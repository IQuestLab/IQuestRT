"""Layer-tail arena state."""

import torch


class TailKernels:
    """The arena slots and device scalars the AR tails share."""

    def __init__(self, hidden: int, eps: float, *, ws, q_len: int):
        self.hidden = hidden
        self.q_len = q_len

        def slot(key, shape, dtype):
            return ws.slot(("tail", key), shape, dtype)

        self.h = slot("h", (q_len, hidden), torch.bfloat16)
        self.hn = slot("hn", (q_len, hidden), torch.bfloat16)
        self.out = slot("out", (q_len, hidden), torch.bfloat16)
        # The recursive draft's fp32 residual stream (`fp32_residual_connection`): h after
        # attention, and the layer output after the sandwich tail.
        self.h32 = slot("h32", (q_len, hidden), torch.float32)
        self.out32 = slot("out32", (q_len, hidden), torch.float32)
        self.eps = slot("eps", (1,), torch.float32)
        self.a_scale = slot("a_scale", (1,), torch.float32)
        self.f_scale = slot("f_scale", (1,), torch.float32)
        self._eps_val = float(eps)
        self._eps_set = False
        self._scales = (None, None)

    def _prep(self, a_scale, f_scale):
        if not self._eps_set:
            self.eps.fill_(self._eps_val)
            self._eps_set = True
        # Refilled only on change: inside a captured graph `fill_` bakes the value in, and
        # both scales are constant per model, so this is one host->device write at setup.
        if self._scales != (a_scale, f_scale):
            self.a_scale.fill_(float(a_scale))
            self.f_scale.fill_(float(f_scale))
            self._scales = (a_scale, f_scale)
