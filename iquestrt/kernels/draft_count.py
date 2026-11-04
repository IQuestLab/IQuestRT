"""Device-launched speculative depth and step graphs.

The first three depths always run. Later decisions use only prior drafts, through
the fixed stop-rule tables. The boundary settles caches, records committed output,
and tail-launches the next step until a stop token or token budget is reached."""

import functools
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch
from .sampler import TOP_K, TOP_P

# The stop rule's tables, fit offline.
RULE_PATH = Path(__file__).with_name("stop_rule.json")
RULE_PHI, RULE_KNOTS, RULE_WARPS = 6, 32 * 32, 8
# The step loop's stop ids, at most.
MAX_STOPS = 8

# Declarations only -- `load_inline(functions=...)` generates the pybind module.
_CPP = r"""
#include <torch/extension.h>
#include <vector>
void draft_control_launch(int64_t count, int64_t table, int64_t depth, bool decide,
                          bool loop, int64_t result, int64_t rows, int64_t main_length,
                          int64_t branch_length, int64_t q, int64_t depths,
                          int64_t no_draft, std::vector<int64_t> rule, double keep_above,
                          int64_t floor, bool by_rule, std::vector<int64_t> loop_args);
void draft_decide_launch(int64_t depth, int64_t drafts, std::vector<int64_t> rule,
                         double keep_above, int64_t depths, int64_t floor, int64_t go);
int64_t draft_graph_instantiate(int64_t graph);
void draft_graph_upload(int64_t exec);
void draft_graph_launch(int64_t exec);
void draft_graph_destroy(int64_t exec);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define DRAFT_CHECK(x)                                                              \
  do {                                                                              \
    const cudaError_t err_ = (x);                                                   \
    TORCH_CHECK(err_ == cudaSuccess, #x ": ", cudaGetErrorString(err_));            \
  } while (0)

#define RULE_FULL 0xFFFFFFFFu
#define RULE_TOPK 20
#define RULE_PHI 6
#define RULE_WARPS 8
#define LOOP_STOPS 8

// The stop rule's tables (`StopRule`) and the sampler state it reads.
struct StopRuleArgs {
  const double* weights;     // [2 * depths][RULE_PHI]: a_k's, then a+_{k+1}'s
  const double* knots;       // every map's, concatenated
  const double* values;
  const int* offsets;        // map m = model * depths + k spans [offsets[m], offsets[m+1])
  const double* threshold;   // [depths - 1]
  double* p;                 // [depths]: P_k = a_0... a_k, left by the decision after k
  const long long* q_ids;    // the sampler's [depths][RULE_TOPK]
  const float* q_probs;
  const float* q_logits;
  double keep_above;         // 1 - top_p
  int depths, floor;
};

// One warp's sum, the same on every lane.
__device__ __forceinline__ double rule_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(RULE_FULL, v, o);
  return __shfl_sync(RULE_FULL, v, 0);
}

__device__ __forceinline__ double rule_logit(double u) {
  const double c = fmin(fmax(u, 1e-6), 1.0 - 1e-6);
  return log(c / (1.0 - c));
}

// What one warp's prediction reads from the rule's constant tables: its weights and its
// isotonic map. Nothing in a step writes them, so the control reads them at its entry,
// before its PDL wait (see `StopRule`).
struct RulePre {
  double w[RULE_PHI];
  const double* xp;
  const double* fp;
  int n;
};

// Whether warp `warp` predicts in the decision before depth `depth`, and what: model 0
// (a_k) or 1 (a+_{k+1}) at depth k. Warps 0..n-1 predict a_first.. a_{depth-1} and warp
// n a+_depth, where first = 0 before the floor's depth, else depth - 1 (whose P_{first-1}
// the decision before left in p).
__device__ __forceinline__ bool rule_role(const StopRuleArgs& a, int depth, int warp,
                                          int& model, int& k) {
  const int first = depth == a.floor ? 0 : depth - 1, n = depth - first;
  model = warp < n ? 0 : 1;
  k = warp < n ? first + warp : depth - 1;
  return warp <= n;
}

// This warp's RulePre for the decision before depth `depth`, if it predicts.
__device__ __forceinline__ void rule_pre(const StopRuleArgs& a, int depth, RulePre& r) {
  int model, k;
  if (!rule_role(a, depth, threadIdx.x >> 5, model, k)) return;
  const int m = model * a.depths + k;
  const double* w = a.weights + m * RULE_PHI;
#pragma unroll
  for (int i = 0; i < RULE_PHI; ++i) r.w[i] = w[i];
  const int lo = a.offsets[m];
  r.n = a.offsets[m + 1] - lo;
  r.xp = a.knots + lo;
  r.fp = a.values + lo;
}

// np.interp(x, r's knots, r's values), by one warp: two coalesced passes find the last
// knot <= x (at most 32 * 32 knots), then np.interp's own arithmetic. The first pass's
// knots and values (lane l: knot l * stride, as x1 and f1) come read before x is known;
// the interval's ends are shuffled out of the second pass's lanes, or past its last lane
// out of the first pass's next one.
__device__ double rule_interp(const RulePre& r, int stride, double x1, double f1,
                              double x) {
  const int lane = threadIdx.x & 31;
  const int n = r.n;
  const int b = __popc(__ballot_sync(RULE_FULL, lane * stride < n && x1 <= x)) - 1;
  if (b < 0) return __shfl_sync(RULE_FULL, f1, 0);
  const int i2 = b * stride + lane;
  const bool in2 = lane < stride && i2 < n;
  const double x2 = in2 ? r.xp[i2] : 0.0;
  const double f2 = in2 ? r.fp[i2] : 0.0;
  const int i = b * stride - 1 + __popc(__ballot_sync(RULE_FULL, in2 && x2 <= x));
  const int l = i - b * stride;              // >= 0: knot b * stride is <= x
  const double xi = __shfl_sync(RULE_FULL, x2, l);
  const double fi = __shfl_sync(RULE_FULL, f2, l);
  const double xs = __shfl_sync(RULE_FULL, x2, (l + 1) & 31);
  const double fs = __shfl_sync(RULE_FULL, f2, (l + 1) & 31);
  const double xb = __shfl_sync(RULE_FULL, x1, (b + 1) & 31);
  const double fb = __shfl_sync(RULE_FULL, f1, (b + 1) & 31);
  if (i == n - 1) return fi;
  const bool past = l + 1 == stride;         // knot i + 1 is (b + 1) * stride
  const double xj = past ? xb : xs, fj = past ? fb : fs;
  if (xi == x) return fi;
  const double slope = (fj - fi) / (xj - xi);
  double v = __dadd_rn(__dmul_rn(slope, x - xi), fi);
  if (isnan(v)) {
    v = __dadd_rn(__dmul_rn(slope, x - xj), fj);
    if (isnan(v) && fi == fj) v = fi;
  }
  return v;
}

// The prediction over r from depth k's features and its draft d, by one warp, lane l < 20
// holding the depth's l-th entry; every lane returns it.
__device__ double rule_predict(const StopRuleArgs& a, const RulePre& r, int k, long long d) {
  const int lane = threadIdx.x & 31;
  const bool in = lane < RULE_TOPK;
  const int at = k * RULE_TOPK + (in ? lane : 0);
  const double v = in ? (double)a.q_logits[at] : 0.0;
  const double q = in ? (double)a.q_probs[at] : 0.0;
  const long long id = in ? a.q_ids[at] : -1;
  const int stride = (r.n + 31) / 32;
  const int i1 = lane * stride;
  const double x1 = i1 < r.n ? r.xp[i1] : 0.0;
  const double f1 = i1 < r.n ? r.fp[i1] : 0.0;
  const double v0 = __shfl_sync(RULE_FULL, v, 0);     // every lane: the mask is full
  const double e = in ? exp(v - v0) : 0.0;
  const double p = e / rule_sum(e);
  double tail = p;                           // own mass plus every smaller one's
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double u = __shfl_down_sync(RULE_FULL, tail, o);
    if (lane + o < 32) tail += u;
  }
  const bool keep = in && (lane == 0 || tail > a.keep_above);
  const double mass = (double)(float)rule_sum(keep ? p : 0.0);    // as the log stored it
  const double count = (double)__popc(__ballot_sync(RULE_FULL, keep));
  const double qd = rule_sum(in && id == d ? q : 0.0);
  const double top = __shfl_sync(RULE_FULL, q, 0);
  const double ent = -rule_sum(in && q > 0.0 ? q * log(q) : 0.0);
  const double* w = r.w;
  const double z = w[0] + rule_logit(qd) * w[1] + rule_logit(top) * w[2] +
                   log(fmin(fmax(mass, 1e-6), 1.0)) * w[3] + log(count) * w[4] + ent * w[5];
  return rule_interp(r, stride, x1, f1, 1.0 / (1.0 + exp(-z)));
}

// Whether depth `depth` runs (floor <= depth < depths), from depths 0..depth-1 and their
// drafts: each predicting warp over its `pre` (rule_role), thread 0 against `threshold`,
// threshold[depth - 1], both read ahead. The product the decision before left is read
// with the predictions' own reads. Every thread of the block must call it, and gets the
// decision.
__device__ bool rule_go(const StopRuleArgs& a, int depth, const long long* drafts,
                        const RulePre& pre, double threshold) {
  __shared__ double s_pred[RULE_WARPS];
  __shared__ int s_go;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int first = depth == a.floor ? 0 : depth - 1, n = depth - first;
  const double before = threadIdx.x == 0 && first ? a.p[first - 1] : 0.0;
  int model, k;
  if (rule_role(a, depth, warp, model, k)) {
    const double pred = rule_predict(a, pre, k, drafts[k]);
    if (lane == 0) s_pred[warp] = pred;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double P = first ? before * s_pred[0] : s_pred[0];
    for (int w = 1; w < n; ++w) P *= s_pred[w];
    a.p[depth - 1] = P;
    s_go = !(P * s_pred[n] < threshold);
  }
  __syncthreads();
  return s_go;
}

// The step loop (`DraftControl.run`).
struct LoopArgs {
  long long* state;               // tokens left, steps logged, stop count, the stop ids
  long long* log;                 // [max_steps][3 + depths]: each step's result
  const cudaGraphExec_t* steps;   // [q + 1]: the host graph of each verify width
  long long max_steps;
};

// What a step's end reads of the loop at its entry, before its PDL wait: only the step
// before it wrote the state, and nothing in a step writes the graphs.
struct LoopPre {
  long long left, logged, stops[LOOP_STOPS];
  int n_stops;
  cudaGraphExec_t next;           // the host graph of width depth + 1
};

__device__ __forceinline__ void loop_pre(const LoopArgs& loop, int depth, LoopPre& p) {
  p.left = loop.state[0];
  p.logged = loop.state[1];
  p.n_stops = static_cast<int>(loop.state[2]);
#pragma unroll
  for (int i = 0; i < LOOP_STOPS; ++i) p.stops[i] = loop.state[3 + i];
  p.next = loop.steps[depth + 1];
}

// The boundary of a step that made r = depth drafts: depths 0..depth-1 have run. With
// LOOP and tokens left at the step's start, also the step loop's stop, log and state, and
// thread 0 launches the next step's host graph if the loop goes on. Every thread of the
// block must call it. Each thread reads all it needs in one round; with LOOP the threads
// then decide at one barrier, and thread 0 launches before any of them writes: the
// launched graph starts once this one has completed, so it sees the writes. The verify's
// input row, whose lanes 1..n hold the step's accepted drafts, is the only value read
// that the boundary overwrites.
template <bool LOOP>
__device__ __forceinline__ void draft_boundary(int depth, long long* __restrict__ result,
                                               long long* __restrict__ rows,
                                               long long* __restrict__ main_length,
                                               long long* __restrict__ branch_length,
                                               int q, int depths, long long no_draft,
                                               const LoopArgs& loop, const LoopPre& lp) {
  const int tx = threadIdx.x;
  const int live = depth + 1;    // the committed token and the r = depth drafts
  const int width = 3 + depths;  // a result's length, and a log row's
  const bool in_loop = LOOP && lp.left > 0;
  const long long n = result[0];
  const long long token = tx < q ? result[1 + min(tx, live - 1)] : 0;
  const long long main = tx == 0 ? main_length[0] : 0;
  const long long branch = tx == 0 ? branch_length[0] : 0;
  // The step committed the accepted drafts rows[1..n] and y; its log row is its result.
  const long long committed = in_loop && tx < q ? (tx ? rows[tx] : result[1]) : 0;
  const long long logged = in_loop && tx < width - 1 ? result[tx] : depth;
  const long long left = lp.left - (n + 1);
  if constexpr (LOOP) {          // without LOOP the kernel holds no graph launch
    if (in_loop) {
      bool stop = false;
      if (tx <= n)
        for (int i = 0; i < lp.n_stops; ++i) stop |= committed == lp.stops[i];
      stop = __syncthreads_or(stop);
      if (tx == 0 && !stop && left > 0 && lp.logged + 1 < loop.max_steps && lp.next)
        cudaGraphLaunch(lp.next, cudaStreamGraphTailLaunch);
      if (tx < width) loop.log[lp.logged * width + tx] = logged;
    }
  }
  if (tx < q) {
    rows[tx] = token;
    rows[q + tx] = tx < live ? token : no_draft;
  }
  if (tx == 0) {
    main_length[0] = main + n - static_cast<long long>(q - 1);
    branch_length[0] = branch - static_cast<long long>(depth - 1);
    result[2 + depths] = depth;
    if (in_loop) {
      loop.state[0] = left;
      loop.state[1] = lp.logged + 1;
    }
  }
}

// The step's end for r = depth drafts, with no decision. Without LOOP it holds no graph
// launch, so the host can launch it outside a graph too; with LOOP it launches the next
// step's host graph. These kernels and the control launch with PDL: the sampler before
// them writes the drafts they read.
template <bool LOOP>
__global__ void draft_end(int depth, long long* __restrict__ result,
                          long long* __restrict__ rows, long long* __restrict__ main_length,
                          long long* __restrict__ branch_length, int q, int depths,
                          long long no_draft, LoopArgs loop) {
  LoopPre lp = {};
  if constexpr (LOOP) loop_pre(loop, depth, lp);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  draft_boundary<LOOP>(depth, result, rows, main_length, branch_length, q, depths, no_draft,
                       loop, lp);
}

// The control before depth `depth`, inside a graph only: depth `depth`
// runs iff it is one of the `depths` and below the forced count or, `by_rule`, the stop
// rule runs it, and thread 0 tail-launches its graph, table[depth]. Otherwise the step
// ends for r = depth and, in the step loop, launches the next step's host graph. The
// rule's tables, the graphs and the loop's state are read before the PDL wait.
__global__ void draft_control(const int* __restrict__ count,
                              const cudaGraphExec_t* __restrict__ table, int depth,
                              long long* __restrict__ result, long long* __restrict__ rows,
                              long long* __restrict__ main_length,
                              long long* __restrict__ branch_length, int q, int depths,
                              long long no_draft, StopRuleArgs rule, bool by_rule,
                              LoopArgs loop) {
  RulePre pre;
  LoopPre lp;
  double threshold = 0.0;
  cudaGraphExec_t next = nullptr;
  if (by_rule && depth < depths) {
    rule_pre(rule, depth, pre);
    if (threadIdx.x == 0) threshold = rule.threshold[depth - 1];
  }
  if (threadIdx.x == 0 && depth < depths) next = table[depth];
  loop_pre(loop, depth, lp);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (depth < depths && (by_rule ? rule_go(rule, depth, result + 2, pre, threshold)
                                 : depth < count[0])) {
    if (threadIdx.x == 0) cudaGraphLaunch(next, cudaStreamGraphTailLaunch);
    return;
  }
  draft_boundary<true>(depth, result, rows, main_length, branch_length, q, depths, no_draft,
                       loop, lp);
}

// The stop rule's decision before depth `depth`, outside a graph: into go[0].
__global__ void draft_decide(int depth, const long long* __restrict__ drafts,
                             StopRuleArgs rule, int* __restrict__ go) {
  RulePre pre;
  rule_pre(rule, depth, pre);
  const double threshold = threadIdx.x == 0 ? rule.threshold[depth - 1] : 0.0;
  const bool run = rule_go(rule, depth, drafts, pre, threshold);
  if (threadIdx.x == 0) go[0] = run;
}

static StopRuleArgs rule_args(const std::vector<int64_t>& r, double keep_above,
                              int depths, int floor) {
  StopRuleArgs a = {};
  if (r.empty()) return a;
  TORCH_CHECK(r.size() == 9, "the stop rule takes 9 pointers, got ", r.size());
  a.weights = reinterpret_cast<const double*>(r[0]);
  a.knots = reinterpret_cast<const double*>(r[1]);
  a.values = reinterpret_cast<const double*>(r[2]);
  a.offsets = reinterpret_cast<const int*>(r[3]);
  a.threshold = reinterpret_cast<const double*>(r[4]);
  a.p = reinterpret_cast<double*>(r[5]);
  a.q_ids = reinterpret_cast<const long long*>(r[6]);
  a.q_probs = reinterpret_cast<const float*>(r[7]);
  a.q_logits = reinterpret_cast<const float*>(r[8]);
  a.keep_above = keep_above;
  a.depths = depths;
  a.floor = floor;
  return a;
}

void draft_control_launch(int64_t count, int64_t table, int64_t depth, bool decide,
                          bool loop, int64_t result, int64_t rows, int64_t main_length,
                          int64_t branch_length, int64_t q, int64_t depths,
                          int64_t no_draft, std::vector<int64_t> rule, double keep_above,
                          int64_t floor, bool by_rule, std::vector<int64_t> loop_args) {
  const int* c = reinterpret_cast<const int*>(count);
  const cudaGraphExec_t* t = reinterpret_cast<const cudaGraphExec_t*>(table);
  int d = static_cast<int>(depth), qq = static_cast<int>(q), n = static_cast<int>(depths);
  long long* res = reinterpret_cast<long long*>(result);
  long long* rw = reinterpret_cast<long long*>(rows);
  long long* ml = reinterpret_cast<long long*>(main_length);
  long long* bl = reinterpret_cast<long long*>(branch_length);
  long long nd = static_cast<long long>(no_draft);
  StopRuleArgs ra = rule_args(rule, keep_above, n, static_cast<int>(floor));
  TORCH_CHECK(!by_rule || ra.p, "a control by the stop rule needs its tables");
  TORCH_CHECK(loop_args.size() == 4, "the step loop takes 4 arguments, got ",
              loop_args.size());
  LoopArgs la = {reinterpret_cast<long long*>(loop_args[0]),
                 reinterpret_cast<long long*>(loop_args[1]),
                 reinterpret_cast<const cudaGraphExec_t*>(loop_args[2]),
                 static_cast<long long>(loop_args[3])};
  void* control_args[] = {&c, &t, &d, &res, &rw, &ml, &bl, &qq, &n, &nd, &ra, &by_rule,
                          &la};
  void* end_args[] = {&d, &res, &rw, &ml, &bl, &qq, &n, &nd, &la};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  // The rule's decision before the floor's depth takes floor + 1 warps (rule_go).
  cfg.blockDim = dim3(by_rule ? 32 * static_cast<int>(floor + 1) : 32);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const void* kernel = decide ? reinterpret_cast<const void*>(draft_control)
                       : loop ? reinterpret_cast<const void*>(draft_end<true>)
                              : reinterpret_cast<const void*>(draft_end<false>);
  DRAFT_CHECK(cudaLaunchKernelExC(&cfg, kernel, decide ? control_args : end_args));
}

void draft_decide_launch(int64_t depth, int64_t drafts, std::vector<int64_t> rule,
                         double keep_above, int64_t depths, int64_t floor, int64_t go) {
  StopRuleArgs ra = rule_args(rule, keep_above, static_cast<int>(depths),
                              static_cast<int>(floor));
  TORCH_CHECK(ra.p, "a decision needs the stop rule's tables");
  draft_decide<<<1, 32 * static_cast<int>(floor + 1), 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<int>(depth), reinterpret_cast<const long long*>(drafts), ra,
      reinterpret_cast<int*>(go));
  DRAFT_CHECK(cudaGetLastError());
}

int64_t draft_graph_instantiate(int64_t graph) {
  cudaGraphExec_t exec;
  DRAFT_CHECK(cudaGraphInstantiateWithFlags(&exec, reinterpret_cast<cudaGraph_t>(graph),
                                            cudaGraphInstantiateFlagDeviceLaunch));
  return reinterpret_cast<int64_t>(exec);
}

void draft_graph_upload(int64_t exec) {
  DRAFT_CHECK(cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(exec),
                              at::cuda::getCurrentCUDAStream()));
}

void draft_graph_launch(int64_t exec) {
  DRAFT_CHECK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(exec),
                              at::cuda::getCurrentCUDAStream()));
}

void draft_graph_destroy(int64_t exec) {
  DRAFT_CHECK(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)));
}
"""


@functools.lru_cache(None)
def _ext():
    with cuda_arch():
        return load_inline(name="iquestrt_draft_count_v1", cpp_sources=_CPP,
                           cuda_sources=_CUDA,
                           functions=["draft_control_launch", "draft_decide_launch",
                                      "draft_graph_instantiate", "draft_graph_upload",
                                      "draft_graph_launch", "draft_graph_destroy"],
                           extra_cuda_cflags=["-O3"], verbose=False)


def _check(name, t, shape):
    if (not isinstance(t, torch.Tensor) or not t.is_cuda or t.dtype != torch.int64
            or tuple(t.shape) != shape or not t.is_contiguous()):
        raise ValueError(f"{name} must be a contiguous CUDA int64 tensor of shape {shape}, "
                         f"got {getattr(t, 'dtype', type(t))}{list(getattr(t, 'shape', ()))}")


class StopRule:
    """The stop rule's tables on the device, from `path`, and what it decides from: the
    `q_ids`, `q_probs` and `q_logits` [depths, TOP_K] a sampler's draft calls leave,
    which `bind` points at another sampler's. `p` holds the running
    product the decisions pass on."""

    def __init__(self, q_ids, q_probs, q_logits, *, depths: int, floor: int,
                 path: Path = RULE_PATH):
        spec = json.loads(Path(path).read_text())
        now, ahead = spec["now"], spec["next"]
        if spec["min_depths"] != floor or floor + 1 > RULE_WARPS:
            raise ValueError(f"{path} decides from a floor of {spec['min_depths']} depths "
                             f"in at most {RULE_WARPS - 1}, not {floor}")
        if (len(now["weights"]) != depths or len(ahead["weights"]) != depths - 1
                or len(spec["threshold"]) != depths - 1):
            raise ValueError(f"{path} holds {len(now['weights'])} depths, not {depths}")
        maps = [*zip(now["knots"], now["values"]), *zip(ahead["knots"], ahead["values"]),
                ([], [])]                           # model * depths + k; none past the last
        for knots, values in maps[:-1]:
            if (not 0 < len(knots) <= RULE_KNOTS or len(values) != len(knots)
                    or any(b < a for a, b in zip(knots, knots[1:]))):
                raise ValueError(f"{path}: an isotonic map takes 1..{RULE_KNOTS} ascending "
                                 "knots, each with its value")
        self.depths, self.floor = depths, floor
        self.bind(q_ids, q_probs, q_logits)
        f64, dev = torch.float64, q_ids.device
        self.weights = torch.tensor([*now["weights"], *ahead["weights"], [0.0] * RULE_PHI],
                                    dtype=f64, device=dev)
        self.knots = torch.tensor([x for k, _ in maps for x in k], dtype=f64, device=dev)
        self.values = torch.tensor([x for _, v in maps for x in v], dtype=f64, device=dev)
        bounds = [0]
        for knots, _ in maps:
            bounds.append(bounds[-1] + len(knots))
        self.offsets = torch.tensor(bounds, dtype=torch.int32, device=dev)
        self.threshold = torch.tensor(spec["threshold"], dtype=f64, device=dev)
        self.p = torch.zeros(depths, dtype=f64, device=dev)

    def bind(self, q_ids, q_probs, q_logits) -> None:
        """Decide from these buffers, a sampler's, from the next launch or capture on."""
        shape = (self.depths, TOP_K)
        for name, t, dtype in (("q_ids", q_ids, torch.int64), ("q_probs", q_probs,
                               torch.float32), ("q_logits", q_logits, torch.float32)):
            if (not t.is_cuda or t.dtype != dtype or tuple(t.shape) != shape
                    or not t.is_contiguous()):
                raise ValueError(f"{name} must be a contiguous CUDA {dtype} tensor of shape "
                                 f"{shape}")
        self.q_ids, self.q_probs, self.q_logits = q_ids, q_probs, q_logits

    def args(self) -> list[int]:
        """The kernels' pointers, in `StopRuleArgs`' order."""
        return [t.data_ptr() for t in (self.weights, self.knots, self.values, self.offsets,
                                       self.threshold, self.p, self.q_ids, self.q_probs,
                                       self.q_logits)]


class DraftControl:
    """The control kernel of a step of `q` verify rows drafting `floor`..`depths` depths,
    and the depth graphs it launches."""

    def __init__(self, q: int, depths: int, floor: int, no_draft: int,
                 rule: StopRule | None = None, max_steps: int = 0):
        if not 0 < floor <= depths < q:
            raise ValueError(f"a Q={q} verify reads at most {q - 1} drafts, from a floor "
                             f"of at least one: got depths {floor}..{depths}")
        if no_draft >= 0:
            raise ValueError(f"NO_DRAFT must be negative (not a token id), got {no_draft}")
        if rule is not None and (rule.depths, rule.floor) != (depths, floor):
            raise ValueError(f"the stop rule decides depths {rule.floor}..{rule.depths}, "
                             f"not {floor}..{depths}")
        if q > 32 or max_steps < 0:
            raise ValueError(f"one warp ends a step of at most 32 verify rows, logging "
                             f"0 or more steps: got Q={q}, {max_steps}")
        self.q, self.depths, self.floor, self.no_draft = q, depths, floor, no_draft
        self.count = torch.full((1,), depths, dtype=torch.int32, device="cuda")
        self.rule, self.by_rule = rule, False
        self._go = torch.zeros(1, dtype=torch.int32, device="cuda")
        # Depth j's executable at table[j], j = floor..depths-1, read at run time.
        self.table = torch.zeros(depths, dtype=torch.int64, device="cuda")
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, int]] = {}
        # Step loop: its state (the tokens left, the steps logged, the stop count and
        # ids; zero outside `run`), each step's result, and the host graph of verify
        # width w at steps[w].
        i64 = torch.int64
        self.loop = torch.zeros(3 + MAX_STOPS, dtype=i64, device="cuda")
        self.log = torch.zeros((max_steps, 3 + depths), dtype=i64, device="cuda")
        self.steps = torch.zeros(q + 1, dtype=i64, device="cuda")
        self._steps: dict[int, tuple[torch.cuda.CUDAGraph, int]] = {}

    def _rule(self) -> StopRule:
        if self.rule is None:
            raise ValueError("this control has no stop rule to decide by")
        return self.rule

    def _launch(self, depth, decide, loop, result, rows, main_length,
                branch_length) -> None:
        _check("result", result, (3 + self.depths,))
        _check("rows", rows, (2, self.q))
        _check("main_length", main_length, (1,))
        _check("branch_length", branch_length, (1,))
        if (decide or loop) and not torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the control and the step loop's end launch graphs from the "
                               "device, which only a kernel of a captured graph may: call "
                               "them while capturing")
        by_rule = decide and self.by_rule
        _ext().draft_control_launch(
            self.count.data_ptr(), self.table.data_ptr(), depth, decide, loop,
            result.data_ptr(), rows.data_ptr(), main_length.data_ptr(),
            branch_length.data_ptr(), self.q, self.depths, self.no_draft,
            self._rule().args() if by_rule else [], 1.0 - TOP_P, self.floor, by_rule,
            [self.loop.data_ptr(), self.log.data_ptr(), self.steps.data_ptr(),
             self.log.shape[0]])

    def decide(self, depth: int, result) -> bool:
        """The stop rule's decision before depth `depth` from depths 0..depth-1 and their
        drafts in `result`, as the control takes it, on the stream and read back: the
        eager prefill's. Decisions follow one another from the floor's depth, as in a
        step, since each passes the next its running product."""
        rule = self._rule()
        if not self.floor <= depth < self.depths:
            raise ValueError(f"the rule decides depths {self.floor}..{self.depths - 1}, "
                             f"got {depth}")
        _check("result", result, (3 + self.depths,))
        _ext().draft_decide_launch(depth, result[2:].data_ptr(), rule.args(), 1.0 - TOP_P,
                                   self.depths, self.floor, self._go.data_ptr())
        return bool(self._go.item())

    def control(self, depth: int, result, rows, main_length, branch_length) -> None:
        """The control before depth `depth`, inside a captured graph: depth `depth`'s
        graph if it runs, else the step's end for `depth` drafts (`end` with `loop`).
        Before depth `depths` it always ends the step."""
        if not self.floor <= depth <= self.depths:
            raise ValueError(f"a control runs before depths {self.floor}..{self.depths}, "
                             f"got {depth}")
        self._launch(depth, True, True, result, rows, main_length, branch_length)

    def end(self, drafts: int, result, rows, main_length, branch_length, *,
            loop: bool = False) -> None:
        """The step's end for `drafts` drafts, with no decision: the next verify's rows,
        both rewinds and the count into `result` (see `DraftControl`). With `loop`, inside a
        captured graph, it also takes the step loop's stop, log and launch; without, its
        kernel holds no graph launch, so it also runs outside a graph."""
        if not self.floor <= drafts <= self.depths:
            raise ValueError(f"a step drafts {self.floor}..{self.depths}, got {drafts}")
        self._launch(drafts, False, loop, result, rows, main_length, branch_length)

    def capture_depth(self, depth: int, fn) -> None:
        """Capture `fn` -- depth `depth`, then the control before the next -- as the graph
        the control before depth `depth` launches."""
        if not self.floor <= depth < self.depths:
            raise ValueError(f"depth graphs are depths {self.floor}..{self.depths - 1}, "
                             f"got {depth}")
        self._drop(self._graphs, depth)
        graph = torch.cuda.CUDAGraph(keep_graph=True)   # torch never instantiates it
        with torch.cuda.graph(graph):
            fn()
        ext = _ext()
        exe = ext.draft_graph_instantiate(graph.raw_cuda_graph())
        self._graphs[depth] = (graph, exe)
        ext.draft_graph_upload(exe)
        self.table[depth] = exe

    def add_step(self, width: int, graph: torch.cuda.CUDAGraph) -> None:
        """Make `graph`, captured with `keep_graph=True`, the host graph of verify width
        `width` that the step loop launches: instantiated for device launch and uploaded.
        Torch's `replay` instantiates a copy of its own, which runs one step."""
        if not self.floor + 1 <= width <= self.depths + 1:
            raise ValueError(f"a step verifies {self.floor + 1}..{self.depths + 1} rows, "
                             f"got {width}")
        self._drop(self._steps, width)
        ext = _ext()
        exe = ext.draft_graph_instantiate(graph.raw_cuda_graph())
        self._steps[width] = (graph, exe)
        ext.draft_graph_upload(exe)
        self.steps[width] = exe

    def run(self, width: int, tokens: int, stops) -> list[list[int]]:
        """The step loop from the host graph of `width`: each step's end launches the
        next, through the step that commits one of `stops` or brings the tokens the steps
        committed to `tokens`. Returns each step's result [n, y, d_0.. d_{depths-1}, r],
        read once the chain has ended, and leaves the state cleared, so a replay outside
        `run` is one step."""
        stops = [int(t) for t in stops]
        if not 0 < tokens <= self.log.shape[0]:
            raise ValueError(f"a run logs at most {self.log.shape[0]} steps, so it takes "
                             f"1..{self.log.shape[0]} tokens, got {tokens}")
        if len(stops) > MAX_STOPS:
            raise ValueError(f"the step loop takes at most {MAX_STOPS} stop ids, got "
                             f"{len(stops)}")
        if width not in self._steps:
            raise ValueError(f"no host graph of width {width} was added: {sorted(self._steps)}")
        state = [tokens, 0, len(stops), *stops, *[0] * (MAX_STOPS - len(stops))]
        self.loop.copy_(torch.tensor(state, dtype=torch.int64))
        _ext().draft_graph_launch(self._steps[width][1])
        logged = int(self.loop[1])          # the stream's next work follows the chain
        rows = self.log[:logged].tolist()
        self.loop.zero_()
        return rows

    def _drop(self, held: dict, key: int) -> None:
        graph = held.pop(key, None)
        if graph is not None:
            torch.cuda.synchronize()       # no replay may still launch it
            _ext().draft_graph_destroy(graph[1])

    def release(self) -> None:
        """Destroy every depth and host graph's executable: the next request captures
        its own."""
        for held in (self._graphs, self._steps):
            for key in list(held):
                self._drop(held, key)
        self.table.zero_()
        self.steps.zero_()
