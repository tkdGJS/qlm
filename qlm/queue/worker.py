import requests
import uuid
import time
from openai import OpenAI
from qlm.endpoints.endpoint import Endpoint
from transformers import AutoTokenizer

#[SH] 토큰 개수 로컬에서 계산
_TOKENIZER_CACHE = {}

def get_tokenizer(model_name: str):
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _TOKENIZER_CACHE[model_name] = tok
    return tok
####여기까지

INF = float("inf")
class Worker:
    """
    Worker class that represents a single instance of vLLM in the system.
    """

    def __init__(self, address, port, endpoint):
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
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.worker_id = uuid.uuid4()

        # === [추가할 코드] =======================================
        self.total_violation = 0.0  # 누적 위반 시간을 저장할 변수
        self.processed_count = 0  # 전체 처리 횟수
        self.success_count = 0    # SLO 달성 성공 횟수 (Violation == 0)
        # ==================================================

        print(f"Worker {self.worker_id} registered at {self.address}")

    def add_request(self, prompt, model,slo,insertion_time,original_slo,original_insertion_time, max_tokens=None, seq_no=None):
        """
        Add a request to the worker.
        :param prompt: The prompt to be added.
        :param model: The model to be used.
        """

        if self.endpoint.model != model:
            self.endpoint.model_swap(model)

        try:
            #===============================================
            start_time=time.time() # 추가: 요청 별 처리 시작 시간
            first_token_time = None
            #===============================================

            # User prompt => 출력 토큰 계산 및 생성
            kwargs = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

            #completion = self.client.completions.create(model=model, prompt=prompt, **kwargs)

            #[SH] TTFT 추적 로직
            stream = self.client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            **kwargs
            )
            # 스트림 소비 (출력 텍스트가 필요 없으면 누적 안 해도 됨)
            text_parts = []
#            last_usage = None

            for chunk in stream:
                # 텍스트 조각이 실제로 올 때 TTFT를 찍는 게 더 정확함
                if getattr(chunk, "choices", None):
                    t = getattr(chunk.choices[0], "text", None)
                    if t:
                        if first_token_time is None:
                            first_token_time = time.time()
                        text_parts.append(t)

#                u = getattr(chunk, "usage", None)
#                if u is not None:
#                    last_usage = u

            # TTFT
            ttft = (first_token_time - start_time) if first_token_time else None
            #[SH] TTFT 추적 로직 끝

            end_time = time.time()                    # 1. 실제 종료 시간

            # === [추가된 부분: SLO 위반 계산 로직] ===
            #=================================================================================================================
            deadline = original_insertion_time + original_slo  # 2. 마감 기한
            diff = end_time - deadline                # 3. 차이 (양수면 지각, 음수면 여유)
            wait_time=start_time - original_insertion_time     # 4. 큐 대기 시간
            latency=end_time - start_time             # 5. 요청 처리 시간
            violation = max(0, diff)                  # 6. violation 계산
            
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
            output_text = "".join(text_parts)
            tok = get_tokenizer(model)
            prompt_toks = len(tok.encode(prompt, add_special_tokens=False))
            completion_toks = len(tok.encode(output_text, add_special_tokens=False))
            total_toks = prompt_toks + completion_toks
            # ---------------------------


            # [로그 메시지 생성]
            log_message = (
                f"[DEBUG] OrigSLO: {original_slo:.2f} | "
                #f"[DEBUG] req_id={request_id}"
                f"Insertion Time: {original_insertion_time:.4f} | "
                f"Wait Time: {wait_time:.4f} | "
                f"prompt_tok= {prompt_toks} out_tok= {completion_toks} total_tok= {total_toks} | "
                + ttft_part +
                f"Execution Time: {latency:.4f} | "
                f"Diff: {diff:.4f} | "
                f"Violation: {violation:.4f} | "
                f"TotalViolation: {self.total_violation:.4f} | "
                f"Finished Time: {time.time()} | "
                f"SuccessRate: {attainment_ratio:.2f}% ({self.success_count}/{self.processed_count})"
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
