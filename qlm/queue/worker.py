import requests
import uuid
import time
from openai import OpenAI
from qlm.endpoints.endpoint import Endpoint
from transformers import AutoTokenizer
from qlm.queue.monitoring import VLLMHTTPMonitor
import threading
import httpx
import os
import math
#[SH] 토큰 개수 로컬에서 계산
_TOKENIZER_CACHE = {}

def get_tokenizer(model_name: str):
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _TOKENIZER_CACHE[model_name] = tok
    return tok
####여기까지

def _percentile(sorted_vals, p: float):
    # p in [0,1]
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = int(math.ceil(p * n) - 1)   # nearest-rank
    idx = max(0, min(idx, n - 1))
    return sorted_vals[idx]

INF = float("inf")
class Worker:
    """
    Worker class that represents a single instance of vLLM in the system.
    """
#[SH] 수정 전
#    def __init__(self, address, port, endpoint):
    def __init__(self, address, port, endpoint,
                 # 토큰/시퀀스 예산은 "원격 vLLM 실행 인자"라서
                 # 지금 구조에선 orchestrator(너 코드)가 알고 있어야 함.
                 max_num_batched_tokens: int | None = None,
                 max_num_seqs: int | None = None,
                 max_model_len: int | None = None,
                 kv_hard_wm: float = 0.92,
                 kv_soft_wm: float = 0.95,
                 min_free_vram_gb: float = 1.0,
                 gpu_index: int = 0,
                 ):


        """
        Initialize a worker instance. Uses openAI API to communicate with the worker.
        :param address: The address of the worker.
        :param port: The port of the worker.
        """
        #===========================
        #self.execution_logs = []
        #===========================
        self.address = f"http://localhost:{port}"
        self.endpoint= endpoint
        self.openai_api_base = f"{self.address}/v1"
        self.openai_api_key = "EMPTY"
        # === 모니터/예산 설정 ===
        self.metrics_url = f"{self.address}/metrics"
        self.monitor = VLLMHTTPMonitor(metrics_url=self.metrics_url, gpu_index=gpu_index)

        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len

        self.kv_hard_wm = kv_hard_wm
        self.kv_soft_wm = kv_soft_wm
        self.min_free_vram_bytes = int(min_free_vram_gb * 1024**3)

        self._last_snapshot = None

        #[SH]
#        self.client = OpenAI(
#            api_key=self.openai_api_key,
#            base_url=self.openai_api_base,
#        )
        self._client_local = threading.local()
        self._client_timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        self._stream_retries = max(0, int(os.environ.get("QLM_STREAM_RETRIES", "1")))
        self._stream_retry_sleep_s = float(os.environ.get("QLM_STREAM_RETRY_SLEEP_S", "0.5"))
        self.worker_id = uuid.uuid4()

        # === [추가할 코드] =======================================
        self.total_violation = 0.0  # 누적 위반 시간을 저장할 변수
        self.processed_count = 0  # 전체 처리 횟수
        self.success_count = 0    # SLO 달성 성공 횟수 (Violation == 0)
        # ==================================================

        print(f"Worker {self.worker_id} registered at {self.address}")
#[SH]
#Monitoring
    def snapshot(self):
        self._last_snapshot = self.monitor.snapshot()
        return self._last_snapshot

    @staticmethod
    def _get_int_attr(obj, names, default=0) -> int:
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        return int(default)

    def estimate_batch_tokens(self, reqs) -> int:
        """
        보수적인 추정(안전 우선):
        - prompt 길이 + max_tokens 상한을 더함
        - 네 Request 클래스 필드명에 맞춰 후보를 여러 개 둠
        """
        total = 0
        for r in reqs:
            prompt = self._get_int_attr(r, ["prompt_tokens", "prompt_len", "num_prompt_tokens"], 0)
            out_cap = self._get_int_attr(r, ["max_tokens", "max_new_tokens", "output_len_cap"], 0)
            total += prompt + out_cap
        return total

    def can_dispatch(self, batch_reqs, high_priority: bool = False) -> tuple[bool, str]:
        snap = self.snapshot()

        # 1) KV cache gate
        kv = snap.kv_cache_usage_perc
        if kv is not None:
            wm = self.kv_soft_wm if high_priority else self.kv_hard_wm
            if kv >= wm:
                return False, f"kv_high:{kv:.3f}>=wm:{wm:.3f}"

        # 2) VRAM free gate (NVML)
        if snap.vram_total_bytes and snap.vram_used_bytes:
            free = snap.vram_total_bytes - snap.vram_used_bytes
            if free < self.min_free_vram_bytes:
                return False, f"vram_low:{free}B"

        # 3) token budget cap gate
        if self.max_num_batched_tokens is not None:
            est = self.estimate_batch_tokens(batch_reqs)
            if est > self.max_num_batched_tokens:
                return False, f"token_cap:{est}>{self.max_num_batched_tokens}"

        # 4) seq cap gate
        if self.max_num_seqs is not None and len(batch_reqs) > self.max_num_seqs:
            return False, f"seq_cap:{len(batch_reqs)}>{self.max_num_seqs}"

        return True, "ok"
#Monitoring



    def _get_client(self) -> OpenAI:
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
                timeout=self._client_timeout,
            )
            self._client_local.client = client
        return client

    def _reset_client(self) -> None:
        if hasattr(self._client_local, "client"):
            del self._client_local.client

    def add_request(self, prompt, model, insertion_time,original_slo,original_insertion_time, max_tokens=None, slo_type: int = 0):
        """
        Add a request to the worker.
        :param prompt: The prompt to be added.
        :param model: The model to be used.
        """

        if self.endpoint.model != model:
            self.endpoint.model_swap(model)
        #[SH] snapshot
        snap = self.monitor.snapshot()
        try:
            #===============================================
            start_time=time.time() # 추가: 요청 별 처리 시작 시간
            first_token_time = None

            last_token_time = None
            
            # TBT 측정용
            prev_text_time = None
            tbt_samples = []   # per-request TBT samples
            #===============================================

            # User prompt => 출력 토큰 계산 및 생성
            kwargs = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

            #completion = self.client.completions.create(model=model, prompt=prompt, **kwargs)
            output_text = ""
            for attempt in range(self._stream_retries + 1):
                try:
                    first_token_time = None
                    #[SH] TTFT 추적 로직
                    client = self._get_client()
                    stream = client.completions.create(
                    model=model,
                    prompt=prompt,
                    stream=True,
                    **kwargs
                    )
                    # 스트림 소비 (출력 텍스트가 필요 없으면 누적 안 해도 됨)
                    text_parts = []
#                    last_usage = None

                    for chunk in stream:
                        if getattr(chunk, "choices", None):
                            t = getattr(chunk.choices[0], "text", None)
                            if t:
                                now = time.time()
                    
                                if first_token_time is None:
                                    first_token_time = now
                    
                                if prev_text_time is not None:
                                    tbt_samples.append(now - prev_text_time)
                                prev_text_time = now
                    
                                last_token_time = now
                                text_parts.append(t)


#                    u = getattr(chunk, "usage", None)
#                    if u is not None:
#                        last_usage = u

                    output_text = "".join(text_parts)
                    break
                except Exception as stream_error:
                    self._reset_client()
                    if attempt < self._stream_retries:
                        print(f"[Warning] stream error (attempt {attempt + 1}/{self._stream_retries + 1}): {stream_error}")
                        time.sleep(self._stream_retry_sleep_s)
                        continue
                    raise


            # TTFT
            ttft = (first_token_time - start_time) if first_token_time else None

            end_time = time.time() - start_time
            ttlt = end_time - start_time

            # 공통: deadline은 동일하게 계산
            deadline = original_insertion_time + original_slo
            wait_time = start_time - original_insertion_time
            
            # slo_type: TTFT=0, TTLT=1
            if slo_type == 1:  # TTLT
                obs_time = end_time                      # ✅ 완료 관측 시각 = 마지막 토큰 시각(or fallback)
                latency = end_time - start_time          # ✅ TTLT 처리시간
            else:             # TTFT
                obs_time = first_token_time or end_time  # ✅ 첫 토큰 시각(없으면 fallback)
                latency = obs_time - start_time          # ✅ TTFT 처리시간
            
            diff = obs_time - deadline
            violation = max(0.0, diff)

            # 누적 violation 합계 업데이트
            self.total_violation += violation
            
            # SLO attatinment (%)
            self.processed_count +=1
            if violation ==0:
                self.success_count+=1
            attainment_ratio=(self.success_count/self.processed_count)*100 if self.processed_count > 0 else 0.0
            
            #======================================
            #self.execution_logs.append(latency)
#            usage = getattr(chunk, "usage", None)
#            prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
#            completion_toks = getattr(usage, "completion_tokens", None) if usage else None
#            total_toks = getattr(usage, "total_tokens", None) if usage else None
            #======================================
            ttft_part = f"TTFT: {ttft:.4f} | " if ttft is not None else "TTFT: None | "

            # ---- 로컬 토큰 계산 추가 ----
            tok = get_tokenizer(model)
            prompt_toks = len(tok.encode(prompt, add_special_tokens=False))
            completion_toks = len(tok.encode(output_text, add_special_tokens=False))
            total_toks = prompt_toks + completion_toks
            # ---------------------------
            def _fmt(x, fmt_str="{:.4f}", none="NA"):
                return none if x is None else fmt_str.format(x)
            
            def _fmt_bytes_to_gb(b, none="NA"):
                if b is None:
                    return none
                return f"{b / (1024**3):.2f}GiB"
            
            tbt_sorted = sorted(tbt_samples)
            tbt_p95 = _percentile(tbt_sorted, 0.95)
            tbt_p99 = _percentile(tbt_sorted, 0.99)
            
            tbt_part = f"TBT_p95={_fmt(tbt_p95)} | TBT_p99={_fmt(tbt_p99)} | "
            snap_part = (
                f" | KV={_fmt(snap.kv_cache_usage_perc, '{:.3f}')}"
                f" run={_fmt(snap.num_running, '{:.0f}')}"
                f" wait={_fmt(snap.num_waiting, '{:.0f}')}"
                f" swap={_fmt(snap.num_swapped, '{:.0f}')}"
                f" VRAM={_fmt_bytes_to_gb(snap.vram_used_bytes)}/{_fmt_bytes_to_gb(snap.vram_total_bytes)}"
            )

            # [로그 메시지 생성]
            log_message = (
                f"[DEBUG] OrigSLO: {original_slo:.2f} | "
                f"slo_type: {slo_type}"
                #f"[DEBUG] req_id={request_id}"
                f"Insertion Time: {original_insertion_time:.4f} | "
                f"Wait Time: {wait_time:.4f} | "
                f"prompt_tok= {prompt_toks} out_tok= {completion_toks} total_tok= {total_toks} | "
                + ttft_part
                + tbt_part +
                f"TTLT: {ttlt:.4f} | "
                f"Diff: {diff:.4f} | "
                f"Violation: {violation:.4f} | "
                f"TotalViolation: {self.total_violation:.4f} | "
                f"Finished Time: {time.time()} | "
                f"SuccessRate: {attainment_ratio:.2f}% ({self.success_count}/{self.processed_count})"
                + snap_part

            )

            print(log_message)

            try:
                with open("experiment_result_EDF.log", "a", encoding="utf-8") as f:
                    f.write(log_message + "\n")
            except Exception as file_error:
                print(f"[Warning] Failed to write log to file: {file_error}")
            # ==========================================
            
            #print("Result of query:", completion)
            #print(f"[{time.time()}] Result of query:", completion)
        except Exception as e:
            print(f"Error in adding request: {e}")

    def _read_metrics(self, metric_name):
        """
        Reads all metrics from the worker and checks for a match with the metric name.
        """
        metrics = requests.get(f"{self.address}/metrics")

        for line in metrics.text.splitlines():
            if line.startswith(metric_name):
                return float(line.split()[-1])

    def get_backpressure(self):
        """
        Get the backpressure of the worker i.e. the number of requests currently being served.
        Includes running, queued and swapped requests.
        return: The backpressure of the worker.
        """

        try:
            running_requests = self._read_metrics("vllm:num_requests_running")
            queued_requests = self._read_metrics("vllm:num_requests_waiting")
            swapped_requests = self._read_metrics("vllm:num_requests_swapped")

            backpressure = running_requests + queued_requests + swapped_requests

            return backpressure
        except Exception as e:
            # If the worker is not reachable, return infinite backpressure
            return INF

    def __hash__(self):
        return hash(self.worker_id)
