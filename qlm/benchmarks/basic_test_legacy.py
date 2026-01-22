from qlm.queue.queue import Queue
from qlm.endpoints.endpoint import Endpoint
import asyncio
import time
import json
import os
import random

async def basic_test():
    print("=== Basic Benchmark Test for EDF ===")
    random.seed(42)

    endpoint = Endpoint(address="localhost", port=8000, model="meta-llama/Llama-3.2-1B-Instruct")


    print("Waiting for vLLM server to start...")
    time.sleep(10)


    q = Queue() 

    q.register_worker("localhost", 8000, endpoint)


    queue_run_task = asyncio.create_task(q.run_queue())

    queue_run_task.add_done_callback(lambda t: print("run_queue exception:", t.exception()))

    
    #print("\n[Part 1] Warming up with simple prompts...")
    #warmup_prompts = ["Hello", "Hi"]
    #for prompt in warmup_prompts:
    #    q.push(prompt=prompt, model="meta-llama/Llama-3.2-1B-Instruct", insertion_time=time.time(), slo=10)

    # 워밍업 처리 대기
    time.sleep(5)


    dataset_path = "data/ShareGPT_V3_unfiltered_cleaned_split.json"

    if os.path.exists(dataset_path):
        print(f"\n[Part 2] Loading dataset from: {dataset_path}")
        
        with open(dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)


        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        dataset = [(data["conversations"][0]["value"],
                    data["conversations"][1]["value"]) for data in dataset]

        #=== [STEP 1] 실험 규모 설정 ===
        target_count = min(2000, len(dataset)) # 예: 200개 테스트
        print(f"Target Requests: {target_count}")

        # === [STEP 2] 시작 시간 기록 & 로그 파일 초기화 ===
        start_time = time.time()
        start_log_msg = f"!!! BENCHMARK START TIME: {start_time} !!!"
        print(start_log_msg)
        
        # === [STEP 3] 요청 전송 루프 ===
        for i in range(target_count):
            
            current_slo = random.uniform(10,230)//1
            #current_slo = 10
            
            q.push(prompt=dataset[i][0], 
                   model="meta-llama/Llama-3.2-1B-Instruct", 
                   insertion_time=time.time(), 
                   slo=current_slo)
        #[SH]
#        assert False
        print(f"Successfully pushed {target_count} requests.")
        print("Check the worker logs for completion times.")

        # === [STEP 4] 작업 완료 대기 루프 ===
        print("\nWaiting for all requests to be processed...")
        
        while True:
            # 1. 워커들이 현재 처리 중인 작업량 확인
            total_backpressure = 0
            for worker in q.workers:
                total_backpressure += worker.get_backpressure()
            
            # 2. 큐(Virtual Queue)에 대기 중인 그룹이 있는지 확인
            total_queued_groups = 0
            for vq in q.vq_engine.vqs:
                total_queued_groups += len(vq.groups)

            # 상태 출력 (진행 상황 모니터링)
            print(f"Status: [Worker Load: {total_backpressure}] / [Queue Waiting: {total_queued_groups}]")

            # 3. 워커도 놀고 있고, 큐도 비었으면 종료
            if total_backpressure == 0 and total_queued_groups == 0:
                print("All requests processed successfully. Exiting.")
                break
            
            # 5초마다 확인
            await asyncio.sleep(5)


        
    else:
        print(f"Error: Dataset file not found at {dataset_path}")

if __name__ == "__main__":
    asyncio.run(basic_test())
