"""Online attention cells with immutable staging before the PDL wait."""

import functools

from ..attention import ONLINE_GEOMETRY, Q, RING_KV, WINDOW, build_online
from .codegen import compile_transformed, replace_once


# THE BATCHED MAX. TileLang 0.1.15 cannot batch an FP32 max on SM100
# (`attention.build_online`), so each cell reduces the two rows a thread holds one at a
# time: a `__syncthreads()`, the 128-thread exchange through shared memory (two butterfly
# levels of two named barriers each), then the four-lane shuffle -- ten barrier waits a
# CTA. `batch_max` emits both rows' exchanges as one `run_batch` each, as the sums are,
# through the workspace the row sums take next (`workspace_1`, in the K tile's bytes, which
# the QK gemm has read by then): four barrier waits. Per value `run_batch` performs
# `run`'s operations, with the same partners in the same operand order, so every maximum
# and all that follows it is unchanged bit for bit. Its first barrier orders every
# thread's last read of the K tile before any write to the workspace.
#
# The max as TileLang 0.1.15 emits it in every online cell: each thread holds rows r and
# r + 8 of the tile (`i_6`), takes each one's maximum over its four logits, exchanges it
# alone and folds it into the running max.
_ROW_MAX = ("  #pragma unroll\n"
            "  for (int i_6 = 0; i_6 < 2; ++i_6) {\n"
            "    m_clear[i_6] = -CUDART_INF_F;\n"
            "    #pragma unroll\n"
            "    for (int rv = 0; rv < 4; ++rv) {\n"
            "      m_clear[i_6] = max(m_clear[i_6], "
            "sf[((((rv & 1) * 4) + (i_6 * 2)) + (rv >> 1))]);\n"
            "    }\n")
_ROW_EXCHANGE = ("    __syncthreads();\n"
                 "    m_clear[i_6] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>"
                 "::run(m_clear[i_6], (&(((float*)workspace_2)[0])));\n"
                 "    m_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>"
                 "::run(m_clear[i_6]);\n")
_FOLD = "    m[i_6] = max(m[i_6], m_clear[i_6]);\n  }\n"
_BATCH = ("  }\n"
          "  tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>, 2, 128>::run_batch("
          "(&(m_clear[0])), (&(((float*)workspace_1)[0])));\n"
          "  tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>, 2, 128>::run_batch("
          "(&(m_clear[0])));\n"
          "  #pragma unroll\n"
          "  for (int i_6 = 0; i_6 < 2; ++i_6) {\n")
# The row sums' batch: TileLang sizes `workspace_1` for it, both rows 128 floats apart,
# and it takes that workspace after the max.
_SUM_BATCH = ("  tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>, 2, 128>::run_batch("
              "(&(rs[0])), (&(((float*)workspace_1)[0])));\n")


def batch_max(code):
    """Exchange the maxima of the two rows each thread holds in one batch (THE BATCHED
    MAX): close the rows' loop after their local maxima, exchange both, fold both."""
    if code.count(_SUM_BATCH) != 1:
        raise RuntimeError("the cell's row sums no longer batch through workspace_1")
    return replace_once(code, _ROW_MAX + _ROW_EXCHANGE + _FOLD, _ROW_MAX + _BATCH + _FOLD)


@functools.lru_cache(None)
def build_attention_early():
    def transform(code):
        code = batch_max(code)
        begin = code.index("  #pragma unroll\n  for (int i_2 = 0; i_2 < 8; ++i_2) {")
        finish = code.index("  tl::cp_async_commit();", begin) + len("  tl::cp_async_commit();\n")
        segment = code[begin:finish]
        if segment.count("tl::cp_async_gs<16>") != 1:
            raise RuntimeError("Expected one K copy loop")
        code = code[:begin] + code[finish:]
        destination = code.index("  #pragma unroll\n  for (int i = 0; i < 2; ++i) {")
        code = code[:destination] + segment + code[destination:]
        if code.count("tl::cp_async_commit();") != 2:
            raise RuntimeError("Unexpected async-copy group count")
        return code

    return compile_transformed(
        build_online, transform, 2048, bk=64, n_split=32, group_split=3, ring=False
    )


_PDL_PAIR = ("  cudaGridDependencySynchronize();\n"
             "  cudaTriggerProgrammaticLaunchCompletion();\n")
_SCALARS = "  float sc = scale[0];\n  int64_t qpos0 = mask[0];\n"
_COPY = "    tl::cp_async_gs<16>("
_COMMIT = "  tl::cp_async_commit();\n"


def _tile_copy(code, var, tile, source):
    """Cut the loop that copies the CTA's `tile` from `source`; return (code, loop)."""
    head = (f"  #pragma unroll\n  for (int {var} = 0; {var} < 8; ++{var}) {{\n"
            f"{_COPY}(&(((bfloat16_t*){tile})")
    begin = code.index(head)
    end = code.index(_COMMIT, begin) + len(_COMMIT)
    loop = code[begin:end]
    # Copy (thread t, var) moves 8 elements of the tile's row var * 8 + t / 16.
    row = f"(&({source}[(((((int)blockIdx.y) * 8192) + ({var} * 1024)) + (((int)threadIdx.x) * 8))]"
    if code.count(head) != 1 or loop.count(_COPY) != 1 or row not in loop:
        raise RuntimeError("the ring cell's tile copy changed; update its early copy")
    return code[:begin] + code[end:], loop


def ring_transform(code, *, kv):
    """Copy the K/V tiles and read `scale` and `qpos0`, then wait. After the wait, wait
    for those copies, then copy again, with the same instructions, every row whose slot
    is one of this step's: the gemms read the bytes the shipping cell reads. The cell
    reads a ring of `kv` slots."""
    bk = ONLINE_GEOMETRY[kv][0]
    code = replace_once(code, _PDL_PAIR + _SCALARS, "@EARLY@" + _PDL_PAIR + "@AFTER@")
    code, k_loop = _tile_copy(code, "i_2", "k_s", "k")
    code, v_loop = _tile_copy(code, "i_3", "v_s", "v")
    code = code.replace("@EARLY@", _SCALARS + k_loop + v_loop)

    def new_rows(loop, var):
        slot = (f"((((int)blockIdx.y) * {bk}) + ({var} * 8)) + (((int)threadIdx.x) >> 4)")
        if kv & (kv - 1):
            # slot + kv - qpos0 mod kv lies in (0, 2 kv): one remainder is the floor-mod.
            age = f"((({slot}) + {kv} - (((int)qpos0) % {kv})) % {kv})"
        else:
            age = f"((({slot}) - ((int)qpos0)) & {kv - 1})"
        return loop.replace(_COPY, f"    if ({age} < {Q}) tl::cp_async_gs<16>(")

    after = "  tl::cp_async_wait<0>();\n" + new_rows(k_loop, "i_2") + new_rows(v_loop, "i_3")
    code = code.replace("@AFTER@", after)
    # The QK gemm's wait<1> leaves only the V copy in flight; the PV gemm's wait<0> none.
    if code.count("tl::cp_async_wait<1>();") != 1:
        raise RuntimeError("the ring cell's copy waits changed; update its early copy")
    return code


@functools.lru_cache(None)
def build_ring_early(kv=RING_KV):
    """`attention.build_online` for the ring cell reading `kv` slots, its K/V tiles copied
    before its wait: the sliding layers' ring (`RING_KV`, bounded by their window), or
    the 4096 cell."""
    bk, splits, groups, _ = ONLINE_GEOMETRY[kv]
    if (bk, kv // splits) != (64, 64):
        raise RuntimeError("the early copy is written for one 64-slot tile per CTA")

    def transform(code):
        return ring_transform(batch_max(code), kv=kv)

    return compile_transformed(build_online, transform, kv, bk=bk, n_split=splits,
                               group_split=groups, ring=True,
                               window=WINDOW if kv == RING_KV else None)
