from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import time
import httpx

try:
    import pynvml  # pip install nvidia-ml-py3
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# LMCache metric catalog (from LMCache "Metrics Reference" docs).
# We record:
# - Counter/Gauge: a single scalar value
# - Histogram: *_count and *_sum (and optionally buckets if you want to extend)
_LMCACHE_COUNTERS_AND_GAUGES = [
    # Core Request Metrics (counters)
    "lmcache:num_retrieve_requests",
    "lmcache:num_store_requests",
    "lmcache:num_lookup_requests",

    # Token Metrics (counters)
    "lmcache:num_requested_tokens",
    "lmcache:num_hit_tokens",
    "lmcache:num_stored_tokens",
    "lmcache:num_lookup_tokens",
    "lmcache:num_lookup_hits",
    "lmcache:num_vllm_hit_tokens",
    "lmcache:num_prompt_tokens",

    # Hit Rate Metrics
    "lmcache:retrieve_hit_rate",         # gauge
    "lmcache:lookup_hit_rate",           # gauge
    "lmcache:lookup_0_hit_requests",     # counter

    # Cache Usage & Lifecycle Metrics (gauges)
    "lmcache:local_cache_usage",
    "lmcache:remote_cache_usage",
    "lmcache:local_storage_usage",

    # Remote Backend & Network Metrics
    "lmcache:num_remote_read_requests",
    "lmcache:num_remote_read_bytes",
    "lmcache:num_remote_write_requests",
    "lmcache:num_remote_write_bytes",
    "lmcache:remote_ping_latency",       # gauge
    "lmcache:remote_ping_errors",
    "lmcache:remote_ping_successes",
    "lmcache:remote_ping_error_code",    # gauge

    # Local CPU Backend Metrics
    "lmcache:local_cpu_evict_count",
    "lmcache:local_cpu_evict_keys_count",
    "lmcache:local_cpu_evict_failed_count",
    "lmcache:local_cpu_hot_cache_count",         # gauge
    "lmcache:local_cpu_keys_in_request_count",   # gauge

    # Memory Management Metrics
    "lmcache:active_memory_objs_count",           # gauge
    "lmcache:pinned_memory_objs_count",           # gauge
    "lmcache:forced_unpin_count",
    "lmcache:pin_monitor_pinned_objects_count",   # gauge

    # P2P Transfer Metrics
    "lmcache:num_p2p_requests",
    "lmcache:num_p2p_transferred_tokens",

    # Health & Internal System Metrics
    "lmcache:lmcache_is_healthy",                 # gauge
    "lmcache:interval_get_blocking_failed_count", # gauge (per interval)
    "lmcache:kv_msg_queue_size",                  # gauge
    "lmcache:remote_put_task_num",                # gauge
    "lmcache:storage_events_ongoing_count",       # gauge
    "lmcache:storage_events_done_count",          # gauge
    "lmcache:storage_events_not_found_count",     # gauge

    # Chunk Statistics Metrics (gauges)
    "lmcache:chunk_statistics_enabled",
    "lmcache:chunk_statistics_total_requests",
    "lmcache:chunk_statistics_total_chunks",
    "lmcache:chunk_statistics_unique_chunks",
    "lmcache:chunk_statistics_reuse_rate",
    "lmcache:chunk_statistics_bloom_filter_size_mb",
    "lmcache:chunk_statistics_bloom_filter_fill_rate",
    "lmcache:chunk_statistics_file_count",
    "lmcache:chunk_statistics_current_file_size",

    # Connector Metrics (gauges)
    "lmcache:scheduler_unfinished_requests_count",
    "lmcache:connector_load_specs_count",
    "lmcache:connector_request_trackers_count",
    "lmcache:connector_kv_caches_count",
    "lmcache:connector_layerwise_retrievers_count",
    "lmcache:connector_invalid_block_ids_count",
    "lmcache:connector_requests_priority_count",
]

_LMCACHE_HISTOGRAMS = [
    # Hit Rate Metrics
    "lmcache:request_cache_hit_rate",
    # Performance & Latency Metrics
    "lmcache:time_to_retrieve",
    "lmcache:time_to_store",
    "lmcache:time_to_lookup",
    "lmcache:retrieve_speed",
    "lmcache:store_speed",
    # Detailed Profiling Metrics
    "lmcache:retrieve_process_tokens_time",
    "lmcache:retrieve_broadcast_time",
    "lmcache:retrieve_to_gpu_time",
    "lmcache:store_process_tokens_time",
    "lmcache:store_from_gpu_time",
    "lmcache:store_put_time",
    "lmcache:remote_backend_batched_get_blocking_time",
    "lmcache:instrumented_connector_batched_get_time",
    # Cache Usage & Lifecycle Metrics
    "lmcache:request_cache_lifespan",
    # Remote Backend & Network Metrics (docs say ms; still exposed as histogram sum/count)
    "lmcache:remote_time_to_get",
    "lmcache:remote_time_to_put",
    "lmcache:remote_time_to_get_sync",
    # P2P Transfer Metrics
    "lmcache:p2p_time_to_transfer",
    "lmcache:p2p_transfer_speed",
]


@dataclass
class VLLMSnapshot:
    ts: float
    kv_cache_usage_perc: Optional[float]  # 0~1
    num_running: Optional[float]
    num_waiting: Optional[float]
    num_swapped: Optional[float]
    vram_used_bytes: Optional[int]
    vram_total_bytes: Optional[int]

    # KV offloading (누적 카운터: /metrics에서 그대로 가져옴)
    kvo_out_count: Optional[float]   # gpu_to_cpu (swap-out/offload)
    kvo_in_count: Optional[float]    # cpu_to_gpu (swap-in)
    kvo_out_bytes: Optional[float]
    kvo_in_bytes: Optional[float]
    kvo_out_time_s: Optional[float]
    kvo_in_time_s: Optional[float]

    # LMCache metrics (전체를 dict로 수집)
    # - Counter/Gauge: metric_name -> value
    # - Histogram: "<name>_count" and "<name>_sum" keys included when present
    lmcache_metrics: Dict[str, float]


class VLLMHTTPMonitor:
    def __init__(self, metrics_url: str, gpu_index: int = 0, timeout_s: float = 0.5):
        self.metrics_url = metrics_url
        self.gpu_index = gpu_index
        self.timeout_s = timeout_s
        self._nvml_inited = False

    def _nvml_init(self):
        if not _NVML_OK or self._nvml_inited:
            return
        pynvml.nvmlInit()
        self._nvml_inited = True

    def _read_vram(self) -> Tuple[Optional[int], Optional[int]]:
        if not _NVML_OK:
            return None, None
        self._nvml_init()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(mem.used), int(mem.total)

    @staticmethod
    def _parse_gauge(text: str, name: str) -> Optional[float]:
        """
        Prometheus exposition format:
          - name value
          - name{...} value
        """
        # name value
        for line in text.splitlines():
            if line.startswith(name + " "):
                try:
                    return float(line.split()[-1])
                except Exception:
                    return None
        # name{...} value
        for line in text.splitlines():
            if line.startswith(name + "{"):
                try:
                    return float(line.split()[-1])
                except Exception:
                    return None
        return None

    @staticmethod
    def _parse_labeled(text: str, name: str, must_contain: list[str]) -> Optional[float]:
        """
        Prometheus: name{label="...",...} value
        must_contain(예: transfer_type="gpu_to_cpu")가 모두 포함된 라인만 매칭.
        """
        prefix = name + "{"
        for line in text.splitlines():
            if not line.startswith(prefix):
                continue
            if any(s not in line for s in must_contain):
                continue
            try:
                return float(line.split()[-1])
            except Exception:
                return None
        return None

    def _read_lmcache_metrics(self, txt: str) -> Dict[str, float]:
        """
        Collect all LMCache metrics we know about.
        - Counter/Gauge: record scalar value if present.
        - Histogram: record *_count and *_sum if present.
          (Buckets are not collected here; use PromQL histogram_quantile() on *_bucket in Prometheus/Grafana.)
        """
        out: Dict[str, float] = {}

        # Scalars
        for m in _LMCACHE_COUNTERS_AND_GAUGES:
            v = self._parse_gauge(txt, m)
            if v is not None:
                out[m] = v

        # Histograms: capture count/sum (sufficient for sanity checks; quantiles via PromQL on buckets)
        for h in _LMCACHE_HISTOGRAMS:
            c = self._parse_gauge(txt, h + "_count")
            s = self._parse_gauge(txt, h + "_sum")
            if c is not None:
                out[h + "_count"] = c
            if s is not None:
                out[h + "_sum"] = s

        return out

    def _read_metrics(self) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout_s) as c:
                r = c.get(self.metrics_url)
                r.raise_for_status()
                txt = r.text
        except Exception:
            return {
                "kv": None, "running": None, "waiting": None, "swapped": None,
                "kvo_out_count": None, "kvo_in_count": None,
                "kvo_out_bytes": None, "kvo_in_bytes": None,
                "kvo_out_time_s": None, "kvo_in_time_s": None,
                "lmcache_metrics": {},
            }

        kv = self._parse_gauge(txt, "vllm:kv_cache_usage_perc")
        if kv is None:
            kv = self._parse_gauge(txt, "vllm:gpu_cache_usage_perc")

        running = self._parse_gauge(txt, "vllm:num_requests_running")
        waiting = self._parse_gauge(txt, "vllm:num_requests_waiting")
        swapped = self._parse_gauge(txt, "vllm:num_requests_swapped")

        # 0~100으로 나오는 경우를 대비해 정규화
        if kv is not None and kv > 1.0:
            kv = kv / 100.0

        # KV offloading (누적)
        kvo_out_count = self._parse_labeled(txt, "vllm:kv_offload_size_count",
                                            ['transfer_type="gpu_to_cpu"'])
        kvo_in_count  = self._parse_labeled(txt, "vllm:kv_offload_size_count",
                                            ['transfer_type="cpu_to_gpu"'])
        kvo_out_bytes = self._parse_labeled(txt, "vllm:kv_offload_total_bytes",
                                            ['transfer_type="gpu_to_cpu"'])
        kvo_in_bytes  = self._parse_labeled(txt, "vllm:kv_offload_total_bytes",
                                            ['transfer_type="cpu_to_gpu"'])
        kvo_out_time  = self._parse_labeled(txt, "vllm:kv_offload_total_time",
                                            ['transfer_type="gpu_to_cpu"'])
        kvo_in_time   = self._parse_labeled(txt, "vllm:kv_offload_total_time",
                                            ['transfer_type="cpu_to_gpu"'])

        # LMCache metrics (if LMCache is integrated + multiproc configured, they appear in the same /metrics)
        lmcache_metrics = self._read_lmcache_metrics(txt)

        return {
            "kv": kv, "running": running, "waiting": waiting, "swapped": swapped,
            "kvo_out_count": kvo_out_count, "kvo_in_count": kvo_in_count,
            "kvo_out_bytes": kvo_out_bytes, "kvo_in_bytes": kvo_in_bytes,
            "kvo_out_time_s": kvo_out_time, "kvo_in_time_s": kvo_in_time,
            "lmcache_metrics": lmcache_metrics,
        }

    def snapshot(self) -> VLLMSnapshot:
        m = self._read_metrics()
        used, total = self._read_vram()
        return VLLMSnapshot(
            ts=time.time(),
            kv_cache_usage_perc=m["kv"],
            num_running=m["running"],
            num_waiting=m["waiting"],
            num_swapped=m["swapped"],
            vram_used_bytes=used,
            vram_total_bytes=total,
            kvo_out_count=m["kvo_out_count"],
            kvo_in_count=m["kvo_in_count"],
            kvo_out_bytes=m["kvo_out_bytes"],
            kvo_in_bytes=m["kvo_in_bytes"],
            kvo_out_time_s=m["kvo_out_time_s"],
            kvo_in_time_s=m["kvo_in_time_s"],
            lmcache_metrics=m["lmcache_metrics"],
        )
