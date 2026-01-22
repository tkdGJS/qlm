from qlm.queue.queue import Queue
from qlm.endpoints.endpoint import Endpoint
import asyncio
import time
import json
import os
import random
from transformers import AutoTokenizer
from pathlib import Path

def read_vllm_max_model_len_from_start_sh() -> int | None:
    # Endpoint가 start_vllm.sh를 실행하므로, 그 값을 "소스 오브 트루스"로 쓰자.
    proj_dir = os.environ.get("QLMPROJDIR")
    if not proj_dir:
        return None
    sh_path = Path(proj_dir) / "qlm" / "endpoints" / "start_vllm.sh"
    if not sh_path.exists():
        return None

    txt = sh_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"--max-model-len\s+(\d+)", txt)
    return int(m.group(1)) if m else None

def dump_jsonl(path: str, rows: list[dict], limit: int | None = None):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if limit is not None and i >= limit:
                break
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def summarize_lengths(name: str, lens: list[int]):
    if not lens:
        print(f"{name}: empty")
        return
    s = sorted(lens)
    def pct(p: float) -> int:
        return s[int(p * (len(s) - 1))]
    print(
        f"{name}: n={len(s)} | min={s[0]} p50={pct(0.50)} p90={pct(0.90)} "
        f"p99={pct(0.99)} max={s[-1]}"
    )
    
async def basic_test():

    def n_prompt_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
        
    print("=== Basic Benchmark Test for EDF ===")
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True
    )

    endpoint = Endpoint(address="localhost", port=8000, model="meta-llama/Llama-3.2-1B-Instruct")


    print("Waiting for vLLM server to start...")
    time.sleep(10)


    q = Queue() 

    q.register_worker("localhost", 8000, endpoint)


    queue_run_task = asyncio.create_task(q.run_queue())

    queue_run_task.add_done_callback(lambda t: print("run_queue exception:", t.exception()))

    
    time.sleep(5)


    dataset_path = "data/ShareGPT_V3_unfiltered_cleaned_split.json"

    if os.path.exists(dataset_path):
        print(f"\n[Part 2] Loading dataset from: {dataset_path}")
        
        with open(dataset_path, encoding='utf-8') as f:
            raw = json.load(f)



        # -----------------------------
        # 2) dataset1 후보: 멀티턴을 (human->gpt) 페어로 전부 쪼개기
        #    dataset2 후보: 첫 왕복(0/1)만
        # -----------------------------
        dataset1_candidates = []  # (tok_len, prompt, answer)
        dataset2_candidates = []  # (tok_len, prompt, answer)

        for item in raw:
            conv = item.get("conversations", [])
            if not isinstance(conv, list) or len(conv) < 2:
                continue
        
            # dataset2: 첫 왕복(0/1)만
            a0, b0 = conv[0], conv[1]
            if a0.get("from") == "human" and b0.get("from") == "gpt":
                p = (a0.get("value") or "").strip()
                a = (b0.get("value") or "").strip()
                if p and a:
                    dataset2_candidates.append((n_prompt_tokens(p), p, a))
        
            # dataset1: 멀티턴 전체에서 human->gpt 페어를 전부 뽑기
            for i in range(len(conv) - 1):
                a_i, b_i = conv[i], conv[i + 1]
                if a_i.get("from") == "human" and b_i.get("from") == "gpt":
                    p = (a_i.get("value") or "").strip()
                    a = (b_i.get("value") or "").strip()
                    if p and a:
                        dataset1_candidates.append((n_prompt_tokens(p), p, a))
        
        print(f"dataset1_candidates(pairs from multi-turn): {len(dataset1_candidates)}")
        print(f"dataset2_candidates(first turn only)      : {len(dataset2_candidates)}")
        
        # -----------------------------
        # 3) “짧은/긴” 기준으로 필터 + 부족하면 fallback
        # -----------------------------
        SHORT_MAX_TOKENS = 128     # dataset1: 짧은 프롬프트 기준
        LONG_MIN_TOKENS  = 1024    # dataset2: 긴 프롬프트 기준
        TARGET_COUNT     = 2000    # 각 데이터셋에 넣을 요청 수
        
        # dataset1 = 짧은 프롬프트 위주
        short_only = [x for x in dataset1_candidates if x[0] <= SHORT_MAX_TOKENS]
        short_only.sort(key=lambda x: x[0])  # 짧은 순
        if len(short_only) >= TARGET_COUNT:
            chosen1 = short_only[:TARGET_COUNT]
        else:
            # 짧은 게 부족하면: 전체 후보에서 가장 짧은 것부터 채움
            dataset1_candidates.sort(key=lambda x: x[0])
            chosen1 = dataset1_candidates[:min(TARGET_COUNT, len(dataset1_candidates))]
            print(f"[WARN] short pairs 부족: short_only={len(short_only)} -> fallback to shortest {len(chosen1)}")
        
        dataset1 = [(p, a) for (_, p, a) in chosen1]
        
        # dataset2 = 긴 프롬프트 위주 (첫 왕복만)
        long_only = [x for x in dataset2_candidates if x[0] >= LONG_MIN_TOKENS]
        long_only.sort(key=lambda x: x[0], reverse=True)  # 긴 순
        if len(long_only) >= TARGET_COUNT:
            chosen2 = long_only[:TARGET_COUNT]
        else:
            # 긴 게 부족하면: 첫 왕복 후보에서 가장 긴 것부터 채움
            dataset2_candidates.sort(key=lambda x: x[0], reverse=True)
            chosen2 = dataset2_candidates[:min(TARGET_COUNT, len(dataset2_candidates))]
            print(f"[WARN] long first-turn 부족: long_only={len(long_only)} -> fallback to longest {len(chosen2)}")
        
        dataset2 = [(p, a) for (_, p, a) in chosen2]
        
        # 통계 출력
        if dataset1:
            print(f"dataset1(short multi-pairs): {len(dataset1)} | token range ~ {chosen1[0][0]}..{chosen1[-1][0]}")
        if dataset2:
            print(f"dataset2(long first-pair) : {len(dataset2)} | token range ~ {chosen2[-1][0]}..{chosen2[0][0]}")
        
        # -----------------------------
        # 4) 큐로 push (너가 원한 형태 그대로)
        # -----------------------------
        start_time = time.time()
        print(f"!!! BENCHMARK START TIME: {start_time} !!!")


        # dataset1 push
        target_count1 = min(TARGET_COUNT, len(dataset1))
        print(f"Pushing dataset1={target_count1}")
        for i in range(target_count1):
            current_slo = random.uniform(1, 10) // 1
            q.push(
                prompt=dataset1[i][0],
                model="meta-llama/Llama-3.2-1B-Instruct",
                insertion_time=time.time(),
                slo=current_slo
            )
        
        # dataset2 push
        target_count2 = min(TARGET_COUNT, len(dataset2))
        print(f"Pushing dataset2={target_count2}")
        for i in range(target_count2):
            current_slo = random.uniform(100, 200) // 1
            q.push(
                prompt=dataset2[i][0],
                model="meta-llama/Llama-3.2-1B-Instruct",
                insertion_time=time.time(),
                slo=current_slo
            )


        #[SH]
#assert False
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
