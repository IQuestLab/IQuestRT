"""The decode MoE holder: the block-FP8 Gate+SiLU and Down of `sm100_moe_fp8.py`."""

import torch

from .sm100_moe_fp8 import ROW_BM, build_down_block, build_gate_silu_block


class MoeDecodeKernels:
    """Block-FP8 Gate+SiLU, then block-FP8 Down into a rotating symmetric output.

    Buffers come from the arena once, so every captured graph records fixed addresses."""

    def __init__(self, experts: int, topk: int, hidden: int, i_loc: int, *, ws,
                 q_len: int, live_rows):
        shape = (experts, topk, hidden, i_loc, q_len)
        if shape != (256, 8, 3072, 192, 8):
            raise ValueError(f"the Q8 MoE needs E256/top8/H3072/I192/Q8, got {shape}")
        live_rows = tuple(live_rows)
        if not live_rows or not all(1 < r <= q_len for r in live_rows):
            raise ValueError(f"live_rows must be in 2..{q_len}, got {live_rows}")
        self.q_len = q_len
        # rows -> (Gate, Down): one row, each verify width's live rows, and all Q (the
        # default).
        self._launch = {r: (build_gate_silu_block(r)[0], build_down_block(r)[0])
                        for r in sorted({1, *live_rows, q_len})}
        self.x_pad = ws.slot(("moe", "x_pad"), (ROW_BM, hidden), torch.bfloat16)
        self.x_pad_fp8 = ws.slot(("moe", "x_pad_fp8"), (ROW_BM, hidden),
                                 torch.float8_e4m3fn)
        self.act_fp8 = ws.slot(("moe", "act_fp8"), (q_len * topk, ROW_BM, i_loc),
                               torch.float8_e4m3fn)
        self.outputs = tuple(
            ws.slot(("moe", "out", index), (ROW_BM, hidden), torch.bfloat16, symm=True)
            for index in range(2)
        )
        self._out_index = 0

    def next_out(self) -> torch.Tensor:
        """The next of the two output slots.

        The MLP AR tail's one-input-barrier NVLS consumer needs two alternating producer
        slots. The index advances while graphs are captured, so each captured call
        records a fixed slot and replays keep the alternation."""
        out = self.outputs[self._out_index]
        self._out_index ^= 1
        return out

    def __call__(self, sel, wt, fc, fc_scale, proj, proj_scale, *, rows=None):
        """Routed pairs `sel`/`wt` [Q*topk] -> [Q,H] bf16, unreduced across ranks.

        `fc`/`proj` are the E4M3 expert banks with their rank-local scale tables
        (`weights.LayerWeights`). `x_pad_fp8` is already written by the attention AR
        tail; `sel` is int32 and ascending by expert id within each row. `rows`
        (default Q) computes the first `rows` rows only."""
        rows = self.q_len if rows is None else rows
        if rows not in self._launch:
            raise ValueError(f"no MoE launch for rows={rows}; built {sorted(self._launch)}")
        out = self.next_out()
        gate_up, down = self._launch[rows]
        gate_up(self.x_pad_fp8, fc, fc_scale, sel, self.act_fp8)
        down(self.act_fp8, proj, proj_scale, sel, wt, out)
        return out[:self.q_len]
