"""Anthropic Messages API request -> chat-template messages -> token ids."""

from __future__ import annotations

import re
import secrets
import sys
from array import array
from collections import OrderedDict


BILLING_HEADER = "x-anthropic-billing-header"


def _text_of(content, system: bool = False) -> str:
    """Flatten text blocks, omitting Claude Code's changing system billing metadata."""
    if content is None:
        return ""
    if isinstance(content, str):
        return "" if system and content.startswith(BILLING_HEADER) else content
    parts = []
    for b in content:
        if b.get("type") == "text":
            t = b.get("text", "")
        else:
            continue
        if system and t.startswith(BILLING_HEADER):
            continue
        parts.append(t)
    return "\n".join(parts)


def convert_messages(req: dict) -> tuple[list[dict], list[dict] | None]:
    """Returns (chat messages, tools) in the shape the model's chat template expects."""
    msgs: list[dict] = []
    system = req.get("system")
    sys_text = _text_of(system, system=True) if system else ""
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    pending_ids = []
    seen_ids = set()
    for m in req.get("messages", []):
        role, content = m["role"], m.get("content")
        if pending_ids and (role != "user" or not isinstance(content, list)
                            or not any(b.get("type") == "tool_result" for b in content)):
            raise ValueError("tool_use must be followed immediately by all its tool_result blocks")
        if role == "system":
            t = _text_of(content, system=True)
            if t:
                msgs.append({"role": "system", "content": t})
            continue
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if role == "user":
            texts, tools_out = [], []
            for b in content or []:
                t = b.get("type")
                if t == "tool_result":
                    if texts:
                        raise ValueError("tool_result blocks must precede text in their user message")
                    body = _text_of(b.get("content"))
                    if b.get("is_error"):
                        body = f"[error] {body}"
                    tools_out.append(
                        {"role": "tool", "content": body, "tool_call_id": b.get("tool_use_id", "")}
                    )
                elif t == "text":
                    texts.append(b.get("text", ""))
            if tools_out:
                by_id = {t["tool_call_id"]: t for t in tools_out}
                if len(by_id) != len(tools_out) or set(by_id) != set(pending_ids):
                    raise ValueError("tool_result IDs must match the preceding tool_use blocks")
                # The iquest template omits IDs, so position must preserve the association.
                tools_out = [by_id[tid] for tid in pending_ids]
                pending_ids = []
            msgs.extend(tools_out)
            if texts:
                msgs.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            texts, reasoning, calls = [], [], []
            for b in content or []:
                t = b.get("type")
                if t == "text":
                    texts.append(b.get("text", ""))
                elif t == "thinking":
                    reasoning.append(b.get("thinking", ""))
                elif t == "tool_use":
                    calls.append(
                        {
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": b.get("input") or {},
                            },
                        }
                    )
            out = {"role": "assistant", "content": "\n".join(texts)}
            if reasoning:
                out["reasoning_content"] = "\n".join(reasoning)
            if calls:
                out["tool_calls"] = calls
                pending_ids = [call["id"] for call in calls]
                if len(set(pending_ids)) != len(pending_ids) or seen_ids.intersection(pending_ids):
                    raise ValueError("tool_use IDs must be unique")
                seen_ids.update(pending_ids)
            if out["content"] or calls or reasoning:
                msgs.append(out)
    if pending_ids:
        raise ValueError("tool_use must be followed immediately by all its tool_result blocks")
    tools = None
    if req.get("tools") and (req.get("tool_choice") or {}).get("type") != "none":
        tools = []
        for t in req["tools"]:
            schema = dict(t.get("input_schema") or {"type": "object", "properties": {}})
            schema.setdefault("type", "object")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
        tools = tools or None
    return msgs, tools


def _map_strings(value, fn):
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {_map_strings(k, fn): _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def _trie_pattern(words) -> str:
    """A regex for any of `words`, preferring the longest, nested as a trie: matching
    compares a few characters per position instead of trying every word in turn."""
    root = {}
    for word in sorted(words):
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def pattern(node):
        # Longer words first; "" ends a word at this node.
        branches = [re.escape(char) + pattern(child) for char, child in node.items() if char]
        if "" in node:
            branches.append("")
        return branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"

    return pattern(root)


def _entry_bytes(text: str, ids: array) -> int:
    """Memory of a cache entry: its text, its ids, and about 100 bytes of ordered dict."""
    return sys.getsizeof(text) + sys.getsizeof(ids) + 100


class Renderer:
    # Memory for tokenized prompt pieces kept for reuse by later requests.
    CACHE_BYTES = 16 * 2**20

    def __init__(self, tokenizer):
        self.tok = tokenizer
        added = [t for t in tokenizer.added_tokens_decoder.values() if t.special]
        # Hugging Face cuts text at verbatim special tokens before tokenizing, so pieces
        # between them tokenize independently and the cache below reproduces whole-text ids.
        if any(t.lstrip or t.rstrip or t.normalized or t.single_word for t in added):
            raise ValueError("special tokens must match verbatim, without stripping "
                             "or normalization")
        specials = {t.content for t in added}
        self.special_ids = {t: [tokenizer.convert_tokens_to_ids(t)] for t in specials}
        self.special_pattern = _trie_pattern(specials) if specials else None
        self.specials = re.compile(self.special_pattern) if specials else None
        self.boundaries = re.compile(f"({self.special_pattern})") if specials else None
        self.pieces = OrderedDict()  # piece text -> token ids, least recently used first
        self.cached_bytes = 0

    def render(self, req: dict) -> tuple[str, list[int]]:
        msgs, tools = convert_messages(req)
        if not msgs:
            raise ValueError("messages must contain at least one message after content conversion")
        thinking = req.get("thinking") or {}
        enable_thinking = thinking.get("type", "enabled") != "disabled"
        # Request text may contain special-token literals such as "<|im_end|>". Render
        # them as per-request placeholders, then tokenize them as plain text, so only
        # the chat template itself produces control tokens.
        literals = {}
        tag = "IQRT" + secrets.token_hex(8)

        def shield(s):
            def placeholder(match):
                return literals.setdefault(match.group(0), f"{tag}N{len(literals)}X")
            return self.specials.sub(placeholder, s)

        if self.specials is not None:
            msgs, tools = _map_strings(msgs, shield), _map_strings(tools, shield)
        text = self.tok.apply_chat_template(
            msgs,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        if self.specials is None:
            return text, self.tok.encode(text, add_special_tokens=False)
        marks = {**self.special_ids, **{
            v: self.tok.encode(k, add_special_tokens=False, split_special_tokens=True)
            for k, v in literals.items()}}
        splitter = (re.compile(f"({tag}N\\d+X|{self.special_pattern})") if literals
                    else self.boundaries)
        parts = splitter.split(text)
        pieces = self._encode(parts[0::2])
        ids = []
        for i, part in enumerate(parts):
            ids += marks[part] if i % 2 else pieces[part]
        if literals:
            restore = {v: k for k, v in literals.items()}
            text = re.sub(f"{tag}N\\d+X", lambda m: restore[m.group(0)], text)
        return text, ids

    def _encode(self, texts: list[str]) -> dict:
        """Token ids for each piece: cached ones reused, the rest in one batch call."""
        out, missing = {"": []}, []
        for text in dict.fromkeys(texts):
            if text in self.pieces:
                self.pieces.move_to_end(text)
                out[text] = self.pieces[text]
            elif text:
                missing.append(text)
        if missing:
            for text, ids in zip(missing, self.tok(missing, add_special_tokens=False)["input_ids"]):
                out[text] = ids = array("I", ids)
                size = _entry_bytes(text, ids)
                if size <= self.CACHE_BYTES:
                    self.pieces[text] = ids
                    self.cached_bytes += size
            while self.cached_bytes > self.CACHE_BYTES:
                self.cached_bytes -= _entry_bytes(*self.pieces.popitem(last=False))
        return out

    def string_params(self, req: dict) -> dict[str, set[str]]:
        """String parameters must remain verbatim, even when they look like JSON."""
        out = {}
        for t in req.get("tools") or []:
            props = (t.get("input_schema") or {}).get("properties") or {}
            out[t.get("name", "")] = {
                k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"
            }
        return out


class Detokenizer:
    """Incremental detokenization (HF-style prefix/read offsets, holds back incomplete UTF-8)."""

    def __init__(self, renderer: Renderer):
        self.tok = renderer.tok
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def feed(self, new_ids: list[int]) -> str:
        self.ids.extend(new_ids)
        prefix_text = self.tok.decode(self.ids[self.prefix : self.read], skip_special_tokens=False)
        new_text = self.tok.decode(self.ids[self.prefix :], skip_special_tokens=False)
        if new_text.endswith("�"):
            return ""
        delta = new_text[len(prefix_text) :]
        self.prefix, self.read = self.read, len(self.ids)
        return delta
