import asyncio
from collections import deque
from qlm.config import Config
from qlm.queue.virtual_queue_engine import VirtualQueueEngine
from qlm.queue.worker import Worker
from qlm.queue.request import Request
from qlm.endpoints.endpoint import Endpoint
import time
import itertools
import os

class Queue:
    """
    Queue class is responsible for managing the queue of requests and workers.
    It is responsible for pushing requests to the queue and running the queue.
    """

    def __init__(self):
        """
        Initializes the queue with an empty list of workers, a Config object and a VirtualQueueEngine object.
        """
        self.workers = []
        self.config = Config()
        self.vq_engine = VirtualQueueEngine()
        #[SH] seq_no
        self._seq_counter = itertools.count(0)

        # run_queue 루프 sleep 간격(초). 기본값 0.1
        # 실험: QLM_QUEUE_LOOP_SLEEP=0.05 같은 식으로 조절
        try:
            self.loop_sleep_s = float(os.environ.get("QLM_QUEUE_LOOP_SLEEP", "0.1"))
        except ValueError:
            self.loop_sleep_s = 0.1
        if self.loop_sleep_s < 0:
            self.loop_sleep_s = 0.0

    def register_worker(self, address, port, endpoint):
        """
        Registers a worker with the queue.
        :param address: The address of the worker.
        :param port: The port of the worker.
        """
        worker = Worker(address, port, endpoint)
        self.workers.append(worker)
        self.vq_engine.add_worker(worker)

    def push(self, prompt, model, slo, insertion_time, max_tokens=None):
        """
        Pushes a request to the virtual queue engine.
        :param prompt: The prompt for the request.
        :param model: The model for the request.
        :param slo: The SLO for the request.
        :param insertion_time: The time at which the request was inserted into the queue. Insertion time is only used for SLO calculation and can be updated during request lifetime.
        :[SH]param max_tokens: (optional) max output tokens for generation
        :[SH]param seqno: input sequence number
        """
        #[SH] seq_no
        seq_no = next(self._seq_counter)
        #
        #new_request = Request(
        #    prompt=prompt, model=model, slo=slo, insertion_time=insertion_time
        #)
        new_request = Request(
            prompt=prompt, model=model, slo=slo, insertion_time=insertion_time, max_tokens=max_tokens, seq_no=seq_no
        )

        self.vq_engine.add_request(new_request)

    async def run_queue(self):
        """
        Runs the queue. The queue runs in an infinite loop and continuously interacts with the virtual queue engine.
        If a request is found, the queue checks for backpressure and if the worker can handle the request.
        If the worker can handle the request, the request is popped from the virtual queue engine and added to the worker.
        """
        last_print_time=0
        while True:
            self.vq_engine.reorder_vqs()
            # === [추가된 로직: 1초마다 상태 출력] ===
            current_time = time.time()
            if current_time - last_print_time >= 5.0: # 1초가 지났는지 확인
                # 큐에 요청이 하나라도 있을 때만 출력 (선택 사항)
                has_any_request = any(len(vq.groups) > 0 for vq in self.vq_engine.vqs)
                if has_any_request:
                    self.vq_engine.print_queue_status() # 아까 만든 출력 함수 호출
                
                last_print_time = current_time # 시간 갱신
            # ====================================
            for worker in self.workers:
                try:
                    backpressure = worker.get_backpressure()
                    has_request = self.vq_engine.has_request(worker)

                    if has_request and backpressure < self.config.max_batch_size:
                        request_to_serve = self.vq_engine.pop_request(worker)
                        asyncio.create_task(
                            asyncio.to_thread(
                                worker.add_request,
                                request_to_serve.prompt,
                                request_to_serve.model,

                                #================================================
                                request_to_serve.slo,             # <--- 추가됨
                                request_to_serve.insertion_time,   # <--- 추가됨
                                request_to_serve.original_slo,
                                request_to_serve.original_insertion_time,
                                request_to_serve.max_tokens,  #[SH] 토큰 출력량 최대치 설정
                                seq_no=request_to_serve.seq_no

                                #================================================
                            )
                        )
                except asyncio.CancelledError as e:
                    print("handling cancelled error", e)
#            await asyncio.sleep(0.1)
            await asyncio.sleep(self.loop_sleep_s)

