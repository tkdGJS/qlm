# monitoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import time
import httpx

try:
    import pynvml  # pip install nvidia-ml-py3
    _NVML_OK = True
except Exception:
    _NVML_OK = False


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

    def _read_vram(self) -> tuple[Optional[int], Optional[int]]:
        if not _NVML_OK:
            return None, None
        self._nvml_init()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(mem.used), int(mem.total)

    @staticmethod
    def _parse_gauge(text: str, name: str) -> Optional[float]:
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

    def _read_metrics(self) -> Dict[str, Optional[float]]:
        try:
            with httpx.Client(timeout=self.timeout_s) as c:
                r = c.get(self.metrics_url)
                r.raise_for_status()
                txt = r.text
        except Exception:
#            return {"kv": None, "running": None, "waiting": None, "swapped": None}
            return {
                "kv": None, "running": None, "waiting": None, "swapped": None,
                "kvo_out_count": None, "kvo_in_count": None,
                "kvo_out_bytes": None, "kvo_in_bytes": None,
                "kvo_out_time_s": None, "kvo_in_time_s": None,
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

#        return {"kv": kv, "running": running, "waiting": waiting, "swapped": swapped}
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

        return {
            "kv": kv, "running": running, "waiting": waiting, "swapped": swapped,
            "kvo_out_count": kvo_out_count, "kvo_in_count": kvo_in_count,
            "kvo_out_bytes": kvo_out_bytes, "kvo_in_bytes": kvo_in_bytes,
            "kvo_out_time_s": kvo_out_time, "kvo_in_time_s": kvo_in_time,
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
        )

