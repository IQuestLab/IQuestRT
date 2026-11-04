"""Prefetch predicted Gate weights for sliding layers with at most five live rows.

The checkpoint-matched fold changes cache residency, not routing or model values.
All 64 CTAs must become resident for the selection barrier. Completion means the
prefetches were issued, not that their bytes have arrived."""

import functools
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.cpp_extension import load_inline

from iquestrt.paths import PREFETCH_FOLD_ENV

from ..arch import cuda_arch

H, E, TOPK = 3072, 256, 8
GATE_ROWS = 384                     # an expert's gate and up rows (2 x I_LOC)
CTAS, THREADS = 64, 128
FORK_ROWS = 5                       # the most live rows a step forks at

_CPP = ("void expert_prefetch_launch(int64_t xn, int64_t w_fold, int64_t logits, "
        "int64_t counter, int64_t rows, int64_t topk, int64_t w_fc, int64_t chosen);")

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int H = 3072, E = 256, ROWS = ROWS_VALUE, THREADS = 128, WARPS = THREADS / 32;
constexpr int CTAS = E / WARPS;                        // one warp an expert
constexpr int SLICES = H / (32 * 8);                   // 16-byte slices of a row a lane holds
constexpr int64_t EXPERT_BYTES = 2LL * 192 * H;        // a Gate expert: 384 x 3072 E4M3
constexpr int CHUNK = 65536;
constexpr int CHUNKS = static_cast<int>(EXPERT_BYTES / CHUNK);
static_assert(EXPERT_BYTES % CHUNK == 0, "whole chunks");
static_assert(E % THREADS == 0, "whole passes over the experts");
static_assert(ROWS <= 2 * WARPS, "two rows a warp at most");

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b, float acc) {
  const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(a2[i]), y = __bfloat1622float2(b2[i]);
    acc = fmaf(x.x, y.x, acc);
    acc = fmaf(x.y, y.y, acc);
  }
  return acc;
}

__global__ void __launch_bounds__(THREADS) expert_prefetch(
    const __nv_bfloat16* __restrict__ xn, const __nv_bfloat16* __restrict__ w_fold,
    float* logits, unsigned* counter, int topk, const uint8_t* __restrict__ w_fc,
    int* __restrict__ chosen) {
  __shared__ int flag[E];
  __shared__ int list[E];
  __shared__ int warp_count[WARPS];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int e = blockIdx.x * WARPS + warp;
  // 1. The logits of this warp's expert, its whole row of W' in flight at once.
  const uint4* wr = reinterpret_cast<const uint4*>(w_fold + static_cast<int64_t>(e) * H);
  uint4 wv[SLICES];
#pragma unroll
  for (int s = 0; s < SLICES; ++s) wv[s] = __ldg(wr + lane + 32 * s);
#pragma unroll 2
  for (int r = 0; r < ROWS; ++r) {
    const uint4* row = reinterpret_cast<const uint4*>(xn + r * H);
    uint4 xv[SLICES];
#pragma unroll
    for (int s = 0; s < SLICES; ++s) xv[s] = row[lane + 32 * s];
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < SLICES; ++s) a = dot8(wv[s], xv[s], a);
#pragma unroll
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
    if (lane == 0) __stcg(logits + r * E + e, a);
  }
  for (int i = tid; i < E; i += THREADS) flag[i] = 0;
  // 2. Every CTA's logits are visible before any CTA selects.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const unsigned old = atomicAdd(counter, 1u);
    const unsigned target = (old / CTAS + 1u) * CTAS;
    unsigned now;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(now) : "l"(counter) : "memory");
    } while (static_cast<int>(now - target) < 0);
  }
  __syncthreads();
  // 3. Each live row's top-k (warp w takes rows w and w + WARPS), then their union.
  for (int r = warp; r < ROWS; r += WARPS) {
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __ldcg(logits + r * E + lane * 8 + i);
    for (int t = 0; t < topk; ++t) {
      float bv = -INFINITY;
      int bi = E;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (v[i] > bv) { bv = v[i]; bi = lane * 8 + i; }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (bi >= E) break;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (lane * 8 + i == bi) v[i] = -INFINITY;
      if (lane == 0) flag[bi] = 1;
    }
  }
  __syncthreads();
  // The union in ascending id order, the same in every CTA: an exclusive prefix count.
  int n = 0;
  for (int p = 0; p < E / THREADS; ++p) {
    const int id = p * THREADS + tid;
    const int mine = flag[id];
    const unsigned below = __ballot_sync(0xFFFFFFFFu, mine) & ((1u << lane) - 1u);
    if (lane == 31) warp_count[warp] = __popc(below) + mine;
    __syncthreads();
    int base = n, add = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      if (w < warp) base += warp_count[w];
      add += warp_count[w];
    }
    if (mine) list[base + __popc(below)] = id;
    if (chosen != nullptr && blockIdx.x == 0) chosen[id] = mine;
    n += add;
    __syncthreads();
  }
  const int total = n * CHUNKS;
  for (int c = blockIdx.x + CTAS * tid; c < total; c += CTAS * THREADS) {
    const uint8_t* p = w_fc + static_cast<int64_t>(list[c / CHUNKS]) * EXPERT_BYTES
                       + static_cast<int64_t>(c % CHUNKS) * CHUNK;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p), "r"(CHUNK) : "memory");
  }
}

}  // namespace

void expert_prefetch_launch(int64_t xn, int64_t w_fold, int64_t logits, int64_t counter,
                            int64_t rows, int64_t topk, int64_t w_fc, int64_t chosen) {
  TORCH_CHECK(rows == ROWS, "expert_prefetch is built for ", ROWS, " rows, got ", rows);
  TORCH_CHECK(topk >= 1 && topk <= 32, "topk must be in 1..32, got ", topk);
  TORCH_CHECK((xn | w_fold) % 16 == 0, "xn and w_fold must be 16-byte aligned");
  // The largest carveout: its CTAs never pin an SM's carveout below what the chain's
  // large-shared-memory kernels need.
  static const bool carveout = [] {
    TORCH_CHECK(cudaFuncSetAttribute(expert_prefetch,
                                     cudaFuncAttributePreferredSharedMemoryCarveout, 100)
                == cudaSuccess, "expert_prefetch carveout");
    return true;
  }();
  (void)carveout;
  expert_prefetch<<<CTAS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(xn), reinterpret_cast<const __nv_bfloat16*>(w_fold),
      reinterpret_cast<float*>(logits), reinterpret_cast<unsigned*>(counter),
      static_cast<int>(topk), reinterpret_cast<const uint8_t*>(w_fc),
      reinterpret_cast<int*>(chosen));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "expert_prefetch launch");
}
"""


# The fewest rows the kernel ranks (BELOW SIX LIVE ROWS IT RANKS SIX).
RANKED_ROWS = 6
_ROW_READ = "const uint4* row = reinterpret_cast<const uint4*>(xn + r * H);"


def ranked_rows(rows: int) -> int:
    """The rows the kernel over `rows` live rows computes and ranks."""
    return max(int(rows), RANKED_ROWS)


def source(rows: int) -> str:
    """The kernel over `rows` live rows: from six as shipped; below six it ranks six,
    the rows past the live ones reading row 0 again (BELOW SIX LIVE ROWS IT RANKS SIX)."""
    rows = int(rows)
    src = _CUDA.replace("ROWS_VALUE", str(ranked_rows(rows)), 1)
    if rows < RANKED_ROWS:
        if src.count(_ROW_READ) != 1:
            raise RuntimeError("expert_prefetch's row read changed")
        src = src.replace(_ROW_READ, _ROW_READ.replace("xn + r * H",
                                                       f"xn + (r < {rows} ? r : 0) * H"))
    return src


@functools.lru_cache(None)
def extension(rows: int):
    """The kernel over `rows` live rows (`source`)."""
    version = "v1" if rows >= RANKED_ROWS else "v2"
    with cuda_arch():
        return load_inline(name=f"iquestrt_expert_prefetch_r{rows}_{version}",
                           cpp_sources=_CPP, cuda_sources=source(rows),
                           functions=["expert_prefetch_launch"],
                           extra_cuda_cflags=["-O3"], verbose=False)


def fold_router(router: torch.Tensor, ffn_norm: torch.Tensor) -> torch.Tensor:
    """W'_0 = bf16(router * feed_forward_norm), [E, H] BF16, the base predictor."""
    if router.dtype != torch.bfloat16 or tuple(router.shape) != (E, H):
        raise ValueError(f"the router must be BF16 [{E}, {H}], got {router.dtype}{list(router.shape)}")
    if ffn_norm.dtype != torch.float32 or tuple(ffn_norm.shape) != (H,):
        raise ValueError(f"feed_forward_norm must be fp32 [{H}], got "
                         f"{ffn_norm.dtype}{list(ffn_norm.shape)}")
    return (router.float() * ffn_norm[None, :]).to(torch.bfloat16).contiguous()


FOLD_KEY = "layer.{}"                 # a prefetched layer's W' in a fold file


def base_digest(base: dict) -> str:
    """The sha256 of W'_0 ({layer: [E, H] BF16}, `fold_router`), layer by layer: what a
    fold file names the checkpoint it was fit for by."""
    h = hashlib.sha256()
    for i in sorted(base):
        w = base[i]
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (E, H):
            raise ValueError(f"layer {i}'s W'_0 must be BF16 [{E}, {H}], "
                             f"got {w.dtype}{list(w.shape)}")
        h.update(f"{i}:".encode())
        h.update(w.detach().cpu().contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def check_fold(path, layers) -> str:
    """Validate the header without loading tensors; return the claimed base digest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"prefetch weights are missing at {path}: download the prefetch file for this "
            f"checkpoint revision (see README) and set {PREFETCH_FOLD_ENV} to it")
    with safe_open(str(path), framework="pt") as f:
        keys, meta = set(f.keys()), f.metadata() or {}
        want = {FOLD_KEY.format(i) for i in layers}
        if keys != want:
            raise ValueError(f"{path} holds {sorted(keys - want)[:4]} beyond and lacks "
                             f"{sorted(want - keys)[:4]} of the prefetched layers")
        for key in keys:
            tensor = f.get_slice(key)
            if tensor.get_dtype() != "BF16" or tensor.get_shape() != [E, H]:
                raise ValueError(f"{path}: {key} must be BF16 [{E}, {H}]")
        digest = meta.get("base_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{path}: missing or invalid base_sha256 metadata")
    return digest


def load_fold(path, layers: dict, device) -> dict:
    """Load predictor weights and match their base digest to the loaded checkpoint."""
    digest = check_fold(path, layers)
    base = {i: fold_router(lw.router_decode_bf16, lw.feed_forward_norm)
            for i, lw in layers.items()}
    if digest != base_digest(base):
        raise ValueError(f"{path} was fit for different router weights; download the "
                         "prefetch file published for this checkpoint revision (see README)")
    # Through the host: safetensors reads "cuda" as GPU 0, where torch's .to() reads the
    # current GPU, each rank's own.
    tensors = load_file(str(path))
    fold = {}
    for i in layers:
        w = tensors[FOLD_KEY.format(i)]
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (E, H):
            raise ValueError(f"{path}: layer {i}'s W' is {w.dtype}{list(w.shape)}, "
                             f"not BF16 [{E}, {H}]")
        fold[i] = w.to(device).contiguous()
    return fold


def launch(xn, w_fold, w_fc, logits, counter, *, rows: int, chosen=None) -> None:
    """One `expert_prefetch` on the current stream: `xn` [Q, H] BF16 (its first `rows`
    rows), `w_fold` a predictor [E, H] BF16, `w_fc` the layer's Gate weights [E, 384, H]
    E4M3; `logits` [ranked_rows(rows), E] fp32 and `counter` [1] int32 its scratch;
    `chosen`, if given, [E] int32 receives the union's flags."""
    if xn.dtype != torch.bfloat16 or xn.dim() != 2 or xn.shape[1] != H or xn.shape[0] < rows:
        raise ValueError(f"xn must be BF16 [>={rows}, {H}], got {xn.dtype}{list(xn.shape)}")
    if not xn.is_contiguous() or w_fold.dtype != torch.bfloat16 or tuple(w_fold.shape) != (E, H):
        raise ValueError("xn must be contiguous and w_fold BF16 [E, H]")
    if w_fc.dtype != torch.float8_e4m3fn or tuple(w_fc.shape) != (E, GATE_ROWS, H) \
            or not w_fc.is_contiguous():
        raise ValueError(f"w_fc must be contiguous E4M3 [{E}, {GATE_ROWS}, {H}]")
    ranked = ranked_rows(rows)
    if logits.dtype != torch.float32 or logits.numel() < ranked * E or counter.dtype != torch.int32:
        raise ValueError("logits must be fp32 [ranked_rows(rows), E] and counter int32")
    extension(rows).expert_prefetch_launch(
        xn.data_ptr(), w_fold.data_ptr(), logits.data_ptr(), counter.data_ptr(), ranked, TOPK,
        w_fc.data_ptr(), 0 if chosen is None else chosen.data_ptr())


class ExpertPrefetch:
    """The prefetch of the main layers in `layers` ({index: LayerWeights}) over a step's
    live rows, on its own stream, each predicting from its W' in `fold` ({index: [E, H]
    BF16}, `load_fold`)."""

    def __init__(self, layers: dict, *, fold: dict, widths, device):
        if not layers:
            raise ValueError("no layer to prefetch")
        if sorted(fold) != sorted(layers):
            raise ValueError(f"the fold covers layers {sorted(fold)}, not {sorted(layers)}")
        self.widths, self.device = tuple(widths), torch.device(device)
        self.rows = max(self.widths)
        forked = [w for w in self.widths if w <= FORK_ROWS]
        self.fold = dict(fold)
        self.w_fc = {i: lw.experts_fc for i, lw in layers.items()}
        self.logits = torch.zeros((max(map(ranked_rows, forked), default=0), E),
                                  dtype=torch.float32, device=self.device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.stream = torch.cuda.Stream(self.device)
        self._forked = False
        # Load every forked width's module now: a lazy load inside an eager step, while a
        # NaN-polled tail spins, can block the host until the device idles.
        first = min(layers)
        xn = torch.zeros((self.rows, H), dtype=torch.bfloat16, device=self.device)
        for w in forked:
            launch(xn, self.fold[first], self.w_fc[first], self.logits[:ranked_rows(w)],
                   self.counter, rows=w)
        torch.cuda.synchronize(self.device)

    def fork(self, layer: int, xn: torch.Tensor) -> None:
        if self.rows not in self.widths:
            raise ValueError(f"no prefetch for width {self.rows}; given {self.widths}")
        if self.rows > FORK_ROWS:
            return
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            launch(xn, self.fold[layer], self.w_fc[layer],
                   self.logits[:ranked_rows(self.rows)], self.counter, rows=self.rows)
        self._forked = True

    def join(self) -> None:
        # Only after a fork: during capture, a wait on a stream with nothing captured
        # would join work recorded outside the graph.
        if self._forked:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._forked = False


def prefetched_layers(specs) -> list[int]:
    """The main layers an MLP tail opens (2..n-1, after layers 1..n-2's MoE) that read
    a sliding window: the layers the side branch prefetches for."""
    return [i for i, spec in enumerate(specs)
            if i >= 2 and not spec.is_dense_mlp and spec.sliding_window is not None]
