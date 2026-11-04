"""Incrementally parse thinking, text, and iquest/iquestcoder tool-call markers."""

from __future__ import annotations

import json

THINK_END = "</think>"
CALL_MARKERS = {
    "<iquest_tool_call>": "</iquest_tool_call>",
    "<iquestcoder_tool_call>": "</iquestcoder_tool_call>",
}
KEY_S, KEY_E, VAL_S, VAL_E = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"


def _partial_suffix(buf: str, marker: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of marker."""
    for n in range(min(len(marker) - 1, len(buf)), 0, -1):
        if marker.startswith(buf[-n:]):
            return n
    return 0


def parse_call_body(body: str, string_params: dict[str, set[str]]):
    """-> (name, args) or None when the body does not follow the grammar."""
    i = body.find(KEY_S)
    name = (body if i < 0 else body[:i]).strip()
    if not name:
        return None
    args: dict = {}
    strings = string_params.get(name, set())
    pos = len(body) if i < 0 else i
    while True:
        rest = body[pos:]
        if not rest.strip():
            break
        if not rest.lstrip().startswith(KEY_S):
            return None
        ks = body.find(KEY_S, pos) + len(KEY_S)
        ke = body.find(KEY_E, ks)
        vs = body.find(VAL_S, ke)
        ve = body.find(VAL_E, vs + len(VAL_S)) if vs >= 0 else -1
        if min(ke, vs, ve) < 0 or body[ke + len(KEY_E) : vs].strip():
            return None
        key = body[ks:ke].strip()
        raw = body[vs + len(VAL_S) : ve]
        if key in strings:
            args[key] = raw
        else:
            try:
                value = json.loads(raw.strip())
                # Reject nonfinite numbers and unpaired surrogates, also nested.
                # Preserve the original argument as for other non-JSON values.
                json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
                args[key] = value
            except (ValueError, TypeError, RecursionError):
                args[key] = raw
        pos = ve + len(VAL_E)
    return name, args


class OutputParser:
    def __init__(
        self, string_params: dict[str, set[str]], in_thinking: bool = True,
        *, tools_enabled: bool = True,
    ):
        self.string_params = string_params
        self.tools_enabled = tools_enabled
        self.state = "think" if in_thinking else "text"
        self.buf = ""
        self.call_start = ""
        self.call_end = ""

    def feed(self, text: str, final: bool = False) -> list[tuple]:
        self.buf += text
        events: list[tuple] = []
        while True:
            if self.state == "think":
                i = self.buf.find(THINK_END)
                if i >= 0:
                    if i:
                        events.append(("thinking", self.buf[:i]))
                    self.buf = self.buf[i + len(THINK_END) :]
                    self.state = "text"
                    continue
                hold = 0 if final else _partial_suffix(self.buf, THINK_END)
                emit, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
                if emit:
                    events.append(("thinking", emit))
                return events
            if self.state == "text":
                # Without tools, model tool markers are ordinary output text.
                # In particular, Claude's auxiliary requests may omit tools.
                if not self.tools_enabled:
                    if self.buf:
                        events.append(("text", self.buf))
                        self.buf = ""
                    return events
                matches = [(self.buf.find(m), m) for m in CALL_MARKERS if m in self.buf]
                if matches:
                    i, self.call_start = min(matches)
                    self.call_end = CALL_MARKERS[self.call_start]
                    if i:
                        events.append(("text", self.buf[:i]))
                    self.buf = self.buf[i + len(self.call_start) :]
                    self.state = "tool"
                    continue
                hold = 0 if final else max(_partial_suffix(self.buf, m) for m in CALL_MARKERS)
                emit, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
                if emit:
                    events.append(("text", emit))
                return events
            # tool: buffer until the closing marker
            i = self.buf.find(self.call_end)
            if i < 0:
                if final:
                    events.append(("text", self.call_start + self.buf))
                    self.buf = ""
                return events
            body, self.buf = self.buf[:i], self.buf[i + len(self.call_end) :]
            parsed = parse_call_body(body, self.string_params)
            if parsed is None:
                events.append(("text", self.call_start + body + self.call_end))
            else:
                events.append(("tool", parsed[0], parsed[1]))
            self.state = "text"
