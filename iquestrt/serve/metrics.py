"""Bounded request metadata and token-weighted cache/latency observations."""
from collections import Counter, OrderedDict
import threading


BUCKETS = (.001,.005,.01,.025,.05,.1,.25,.5, 1., 2.5, 5., 10., 30., 60.)
PHASES = ("restore_ms", "target_prefill_ms", "draft_prefill_ms", "snapshot_ms",
          "initial_draft_ms", "capture_ms", "graph_reused")
LATENCIES = ("time_to_first_token", "time_to_first_content", "time_to_first_visible", "time_to_first_tool",
             "queue", "request_parse", "tokenization", "e2e_request_latency")


class Metrics:
    def __init__(self, max_records=16):
        self.max_records = max_records
        self.lock = threading.Lock()
        self.counters = Counter()
        self.recent = OrderedDict()
        self.histograms = {name: [0, 0., [0] * len(BUCKETS)] for name in LATENCIES}

    def _update(self, rid, record):
        self.recent.setdefault(rid, {"request_id": rid}).update(record)
        while len(self.recent) > self.max_records:
            self.recent.popitem(last=False)

    def _observe(self, name, value):
        if value is None:
            return
        hist = self.histograms[name]
        hist[0] += 1
        hist[1] += value
        for i, bound in enumerate(BUCKETS):
            hist[2][i] += value <= bound

    def record_engine(self, job):
        prompt, cached = len(job.prompt_ids), job.cache_read_tokens
        with self.lock:
            self.counters.update(prefix_cache_queries=prompt, prefix_cache_hits=cached,
                                 prefill_tokens=prompt - cached, engine_requests=1,
                                 graph_reuses=int(job.phases.get("graph_reused", False)))
            self._update(job.request_id, {
                "input_tokens": prompt, "cache_read_tokens": cached,
                "prefill_tokens": prompt - cached,
                "queue_ms": (job.started_at - job.queued_at) * 1000,
                "first_token_ms": ((job.first_token_at - job.received_at) * 1000
                                   if job.first_token_at is not None else None),
                **{k: job.phases[k] for k in PHASES if k in job.phases},
            })
            self._observe("time_to_first_token", None if job.first_token_at is None
                          else job.first_token_at - job.received_at)
            self._observe("queue", job.started_at - job.queued_at)

    def record_request(self, job, *, output_tokens, reason, ended):
        content = (job.first_content_at - job.received_at
                   if job.first_content_at is not None else None)
        visible = (job.first_visible_at - job.received_at
                   if job.first_visible_at is not None else None)
        tool = job.first_tool_at - job.received_at if job.first_tool_at is not None else None
        elapsed = ended - job.received_at
        with self.lock:
            if job.request_recorded:
                return
            job.request_recorded = True
            self.counters.update(requests=1, generation_tokens=output_tokens)
            self.counters["finish_" + reason] += 1
            self._observe("time_to_first_content", content)
            self._observe("time_to_first_visible", visible)
            self._observe("time_to_first_tool", tool)
            self._observe("request_parse", job.parsed_at - job.received_at)
            self._observe("tokenization", job.prepared_at - job.parsed_at)
            self._observe("e2e_request_latency", elapsed)
            self._update(job.request_id, {
                "kind": job.kind, "output_tokens": output_tokens, "finish_reason": reason,
                "request_parse_ms": (job.parsed_at - job.received_at) * 1000,
                "tokenization_ms": (job.prepared_at - job.parsed_at) * 1000,
                "first_content_ms": content * 1000 if content is not None else None,
                "first_visible_ms": visible * 1000 if visible is not None else None,
                "first_tool_ms": tool * 1000 if tool is not None else None,
                "total_ms": elapsed * 1000,
            })

    def snapshot(self):
        with self.lock:
            queries = self.counters["prefix_cache_queries"]
            hits = self.counters["prefix_cache_hits"]
            return {
                "requests": self.counters["requests"],
                "engine": {"prompt_tokens": queries, "cached_tokens": hits,
                           "prefill_tokens": self.counters["prefill_tokens"],
                           "prefix_cache_hit_rate": hits / queries if queries else 0.,
                           "graph_reuses": self.counters["graph_reuses"]},
                "recent": [dict(record) for record in self.recent.values()],
            }

    def render(self):
        lines = []
        with self.lock:
            for name in ("prefix_cache_queries", "prefix_cache_hits", "prefill_tokens",
                         "engine_requests", "graph_reuses", "requests", "generation_tokens"):
                metric = f"iquestrt:{name}_total"
                lines.extend((f"# TYPE {metric} counter", f"{metric} {self.counters[name]}"))
            lines.append("# TYPE iquestrt:requests_finished_total counter")
            for reason in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "cancelled", "error"):
                count = self.counters["finish_" + reason]
                lines.append(f'iquestrt:requests_finished_total{{finish_reason="{reason}"}} {count}')
            for name, (count, total, buckets) in self.histograms.items():
                metric = f"iquestrt:{name}_seconds"
                lines.append(f"# TYPE {metric} histogram")
                for bound, value in zip(BUCKETS, buckets):
                    lines.append(f'{metric}_bucket{{le="{bound:g}"}} {value}')
                lines.extend((f'{metric}_bucket{{le="+Inf"}} {count}',
                              f"{metric}_count {count}", f"{metric}_sum {total}"))
        return "\n".join(lines) + "\n"
