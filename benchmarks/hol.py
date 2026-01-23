from qlm.queue.queue import Queue
from qlm.endpoints.endpoint import Endpoint

import asyncio
import contextlib
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


# =========================
# Helpers
# =========================

def dump_jsonl(path: str, rows: List[Dict[str, Any]], limit: int = 200) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows[:limit]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DEBUG] wrote {min(limit, len(rows))} rows -> {path}")


def summarize_lengths(name: str, lens: List[int]) -> None:
    if not lens:
        print(f"[{name}] empty")
        return
    s = sorted(lens)
    def pct(p: float) -> int:
        return s[int(p * (len(s) - 1))]
    print(
        f"[{name}] n={len(s)} min={s[0]} p50={pct(0.50)} p90={pct(0.90)} p99={pct(0.99)} max={s[-1]}"
    )


def read_vllm_max_model_len_from_start_sh() -> Optional[int]:
    """
    하드코딩 피하려고 start_vllm.sh에서 --max-model-len 값을 파싱.
    (프로젝트가 Endpoint -> start_vllm.sh로 vLLM 띄우는 구조라 이게 현실적으로 잘 맞음)
    """
    candidates = []
    proj = os.environ.get("QLMPROJDIR")
    if proj:
        candidates.append(os.path.join(proj, "qlm", "endpoints", "start_vllm.sh"))
    # 실행 위치에 따라 상대경로도 시도
    candidates.append("start_vllm.sh")
    candidates.append(os.path.join("qlm", "endpoints", "start_vllm.sh"))

    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue

        m = re.search(r"--max-model-len\s+(\d+)", txt)
        if m:
            return int(m.group(1))

    return None


def effective_max_len(tokenizer) -> int:
    """
    vLLM max-model-len + tokenizer.model_max_length 중 더 보수적인 값 사용.
    tokenizer.model_max_length가 1e30 같은 '무한대'면 무시.
    """
    vllm_len = read_vllm_max_model_len_from_start_sh()
    tok_len = getattr(tokenizer, "model_max_length", None)
    if tok_len is not None and tok_len > 1_000_000:
        tok_len = None

    if vllm_len is not None and tok_len is not None:
        eff = min(vllm_len, tok_len)
    else:
        eff = vllm_len or tok_len or 32768

    print(f"[LEN] vllm_max_len={vllm_len}, tokenizer.model_max_length={getattr(tokenizer,'model_max_length',None)} -> effective={eff}")
    return int(eff)


def _print_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc:
        print("run_queue exception:", exc)


def token_len(tokenizer, text: str, trunc_max: Optional[int] = None) -> int:
    """
    괴물 프롬프트 때문에 encode 자체가 느려질 수 있어서,
    trunc_max를 주면 max+1까지만 인코딩해서 '너무 긴지'만 빠르게 판단 가능.
    """
    if trunc_max is None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=trunc_max + 1
    )
    return len(ids)


# =========================
# Main
# =========================

async def basic_test():
    print("=== Basic Benchmark Test (PREFIX mode) ===")
    random.seed(42)

    # --------- knobs (원하면 환경변수로 튜닝) ----------
    DATASET_PATH = os.environ.get("DATASET_PATH", "data/ShareGPT_V3_unfiltered_cleaned_split.json")

    TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "500"))
    RUN_SECONDS = int(os.environ.get("RUN_SECONDS", "300"))
    PUSH_INTERVAL_S = float(os.environ.get("PUSH_INTERVAL_S", "0"))
    PUSH_RATE_RPS = float(os.environ.get("PUSH_RATE_RPS", "0"))
    DRAIN_TIMEOUT_S = float(os.environ.get("DRAIN_TIMEOUT_S", "0"))
    PUSH_LOG_LIMIT = int(os.environ.get("PUSH_LOG_LIMIT", "500"))
    # 실행시간 분리 핵심: 출력 길이 상한
    MAX_TOKENS_SHORT = int(os.environ.get("MAX_TOKENS_SHORT", "32"))     # dataset1
    MAX_TOKENS_LONG  = int(os.environ.get("MAX_TOKENS_LONG", "1024"))    # dataset2

    # dataset1/2를 “입력 길이”로도 어느 정도 구분하고 싶으면 아래 기준 사용
    SHORT_MAX_BASE_PROMPT_TOKENS = int(os.environ.get("SHORT_MAX_BASE_PROMPT_TOKENS", "128"))
    SHORT_MIN_BASE_PROMPT_TOKENS = int(os.environ.get("SHORT_MIN_BASE_PROMPT_TOKENS", "16"))  # 너무 1토큰만 모이는 것 방지

    LONG_MIN_BASE_PROMPT_TOKENS  = int(os.environ.get("LONG_MIN_BASE_PROMPT_TOKENS", "1024"))

    # 랜덤 SLO 범위(실험 목적에 맞게 조절)
    SLO_MIN = int(os.environ.get("SLO_MIN", "10"))
    SLO_MAX = int(os.environ.get("SLO_MAX", "230"))

    # push 순서: seq | shuffle | interleave
    ORDER_MODE = os.environ.get("ORDER_MODE", "shuffle").lower()

    # ---------------------------------------------------

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Prefix 방식: “앞에” 지시문 추가
    PREFIX_SHORT = "Answer in one short sentence.\n\n--- USER PROMPT ---\n"
    PREFIX_LONG  = "Write a detailed answer with at least 800 words.\n\n--- USER PROMPT ---\n"

    # 컨텍스트 한계 계산
    eff_len = effective_max_len(tokenizer)
    SAFETY_MARGIN = 8  # 아주 작은 여유
    # 각 dataset은 output max_tokens가 다르므로 입력 한계도 다름
    max_prompt_tokens_short = max(1, eff_len - MAX_TOKENS_SHORT - SAFETY_MARGIN)
    max_prompt_tokens_long  = max(1, eff_len - MAX_TOKENS_LONG  - SAFETY_MARGIN)

    # prefix 토큰 비용도 반영(최종 프롬프트 기준으로 필터링해야 안전)
    prefix_short_tokens = token_len(tokenizer, PREFIX_SHORT)
    prefix_long_tokens  = token_len(tokenizer, PREFIX_LONG)

    # 최종 프롬프트(=prefix+base_prompt)가 들어갈 수 있는 base_prompt 상한
    max_base_prompt_short = max(1, max_prompt_tokens_short - prefix_short_tokens)
    max_base_prompt_long  = max(1, max_prompt_tokens_long  - prefix_long_tokens)

    print(f"[LEN] max_prompt_tokens_short={max_prompt_tokens_short}, prefix_short_tokens={prefix_short_tokens} -> max_base_prompt_short={max_base_prompt_short}")
    print(f"[LEN] max_prompt_tokens_long ={max_prompt_tokens_long},  prefix_long_tokens ={prefix_long_tokens}  -> max_base_prompt_long ={max_base_prompt_long}")

    # vLLM endpoint/queue 준비
    endpoint = Endpoint(address="localhost", port=8000, model=model_name)

    print("Waiting for vLLM server to start...")
    await asyncio.sleep(10)

    #global queue
    q = Queue()
    #virtual queue
    q.register_worker("localhost", 8000, endpoint)

    queue_run_task = asyncio.create_task(q.run_queue())
    queue_run_task.add_done_callback(_print_task_exception)

    await asyncio.sleep(5)  # 워밍업

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        queue_run_task.cancel()
        return

    print(f"\n[Part 2] Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # -----------------------------
    # dataset 후보 만들기 (base_prompt 기준)
    # dataset1: 멀티턴에서 human->gpt 싱글/멀티 페어 (짧은 base_prompt 위주)
    # dataset2: 첫 왕복만 (긴 base_prompt 위주)
    # -----------------------------
    dataset1_candidates: List[Dict[str, Any]] = []
    dataset2_candidates: List[Dict[str, Any]] = []
    too_long_long: List[Dict[str, Any]] = []  # dataset2용 컨텍스트 초과 후보(추적용)

    # 괴물 프롬프트 빠른 판별을 위해 "max_base_prompt + 1"까지만 토큰화
    # (full encode보다 훨씬 빠름)
    trunc_check_short = max_base_prompt_short
    trunc_check_long  = max_base_prompt_long

    for item in raw:
        item_id = item.get("id")
        conv = item.get("conversations", [])
        if not isinstance(conv, list) or len(conv) < 2:
            continue

        # dataset2: 첫 왕복(0/1)만
        a0, b0 = conv[0], conv[1]
        if a0.get("from") == "human" and b0.get("from") == "gpt":
            base_p = (a0.get("value") or "").strip()
            ans = (b0.get("value") or "").strip()
            if base_p and ans:
                # base_prompt 토큰만 우선 체크(빠르게)
                tl = token_len(tokenizer, base_p, trunc_max=trunc_check_long)
                # tl이 max_base_prompt_long+1이면 "초과"임
                if tl <= max_base_prompt_long:
                    dataset2_candidates.append({
                        "id": item_id,
                        "pair": "0-1",
                        "base_tok_len": tl,
                        "base_prompt": base_p,
                        "answer": ans
                    })
                else:
                    too_long_long.append({
                        "id": item_id, "pair": "0-1", "base_tok_len": tl, "base_prompt": base_p
                    })

#        # dataset1: 멀티턴 전체 페어
#        for i in range(len(conv) - 1):
#            a_i, b_i = conv[i], conv[i + 1]
#            if a_i.get("from") == "human" and b_i.get("from") == "gpt":
#                base_p = (a_i.get("value") or "").strip()
#                ans = (b_i.get("value") or "").strip()
#                if base_p and ans:
#                    tl = token_len(tokenizer, base_p, trunc_max=trunc_check_short)
#                    if tl <= max_base_prompt_short:
#                        dataset1_candidates.append({
#                            "id": item_id,
#                            "pair": f"{i}-{i+1}",
#                            "base_tok_len": tl,
#                            "base_prompt": base_p,
#                            "answer": ans
#                        })
#    print(f"dataset1_candidates(multi-turn pairs, base<=short-limit): {len(dataset1_candidates)}")
        # dataset1: 첫 왕복(0/1)만
        if a0.get("from") == "human" and b0.get("from") == "gpt":
            base_p = (a0.get("value") or "").strip()
            ans = (b0.get("value") or "").strip()
            if base_p and ans:
                tl = token_len(tokenizer, base_p, trunc_max=trunc_check_short)
                if tl <= max_base_prompt_short:
                    dataset1_candidates.append({
                        "id": item_id,
                        "pair": "0-1",
                        "base_tok_len": tl,
                        "base_prompt": base_p,
                        "answer": ans
                    })

    print(f"dataset1_candidates(first-pair, base<=short-limit): {len(dataset1_candidates)}")
    #싱글 페어 dataset1

    print(f"dataset2_candidates(first-pair, base<=long-limit)      : {len(dataset2_candidates)}")
    print(f"too_long_long(>long base-limit)                         : {len(too_long_long)}")

    summarize_lengths("dataset1_candidates base tok lens", [x["base_tok_len"] for x in dataset1_candidates])
    summarize_lengths("dataset2_candidates base tok lens", [x["base_tok_len"] for x in dataset2_candidates])

    # 괴물(초과) 샘플 상위 일부 저장
    too_long_sorted = sorted(too_long_long, key=lambda r: r["base_tok_len"], reverse=True)
    dump_jsonl("debug/debug_too_long_for_long_top50.jsonl", too_long_sorted, limit=50)

    # -----------------------------
    # 최종 dataset1/dataset2 선택 (base_tok_len 기준)
    # -----------------------------
    # dataset1: 짧은 base_prompt + multi-turn 분할
    short_pool = [
        x for x in dataset1_candidates
        if SHORT_MIN_BASE_PROMPT_TOKENS <= x["base_tok_len"] <= SHORT_MAX_BASE_PROMPT_TOKENS
    ]
    short_pool.sort(key=lambda r: r["base_tok_len"])
    chosen1 = short_pool[:min(TARGET_COUNT, len(short_pool))]

    # 부족하면 fallback: 가능한 것 중에서 가장 짧은 것부터
    if len(chosen1) < TARGET_COUNT:
        fallback = sorted(dataset1_candidates, key=lambda r: r["base_tok_len"])
        chosen1 = fallback[:min(TARGET_COUNT, len(fallback))]
        print(f"[WARN] dataset1 short_pool 부족 -> fallback shortest {len(chosen1)}")

    # dataset2: 긴 base_prompt + 첫 왕복만
    long_pool = [x for x in dataset2_candidates if x["base_tok_len"] >= LONG_MIN_BASE_PROMPT_TOKENS]
    long_pool.sort(key=lambda r: r["base_tok_len"], reverse=True)
    chosen2 = long_pool[:min(TARGET_COUNT, len(long_pool))]

    # 부족하면 fallback: 가능한 것 중에서 가장 긴 것부터
    if len(chosen2) < TARGET_COUNT:
        fallback = sorted(dataset2_candidates, key=lambda r: r["base_tok_len"], reverse=True)
        chosen2 = fallback[:min(TARGET_COUNT, len(fallback))]
        print(f"[WARN] dataset2 long_pool 부족 -> fallback longest {len(chosen2)}")

    if chosen1:
        print(f"dataset1 chosen: {len(chosen1)} | base_tok_range ~ {chosen1[0]['base_tok_len']}..{chosen1[-1]['base_tok_len']}")
    else:
        print("dataset1 chosen: empty")

    if chosen2:
        print(f"dataset2 chosen: {len(chosen2)} | base_tok_range ~ {chosen2[-1]['base_tok_len']}..{chosen2[0]['base_tok_len']}")
    else:
        print("dataset2 chosen: empty")

    # “실물 확인용” 저장 (base_prompt 기준)
    dump_jsonl("debug/debug_dataset1_sample_base.jsonl", chosen1, limit=50)
    dump_jsonl("debug/debug_dataset2_sample_base.jsonl", chosen2, limit=50)

    # -----------------------------
    # push 전에 prefix를 붙여 "최종 프롬프트" 만들기
    # (이게 prefix 방식)
    # -----------------------------
    def make_final_prompt(base_prompt: str, prefix: str) -> str:
        return prefix + base_prompt

    def sample_slo() -> int:
        return int(random.uniform(SLO_MIN, SLO_MAX) // 1)

    # 실험 기록(나중에 HOL 분석용): 실제로 큐에 들어간 요청을 JSONL로 남김
    pushed_log: List[Dict[str, Any]] = []

    push_interval_s = PUSH_INTERVAL_S
    if push_interval_s <= 0 and PUSH_RATE_RPS > 0:
        push_interval_s = 1.0 / PUSH_RATE_RPS
    # push 순서 만들기
    # - seq: dataset1 다 넣고 dataset2
    # - interleave: 1개씩 번갈아
    # - shuffle: 둘 합쳐서 랜덤 섞기 (보통 HOL 관찰엔 이게 좋음)
    dataset1_items: List[Dict[str, Any]] = []
    dataset2_items: List[Dict[str, Any]] = []
    for r in chosen1:
        dataset1_items.append({
            "dataset": "dataset1",
            "final_prompt": make_final_prompt(r["base_prompt"], PREFIX_SHORT),
            "base_tok_len": r["base_tok_len"],
            "max_tokens": MAX_TOKENS_SHORT
        })
    for r in chosen2:
        dataset2_items.append({
            "dataset": "dataset2",
            "final_prompt": make_final_prompt(r["base_prompt"], PREFIX_LONG),
            "base_tok_len": r["base_tok_len"],
            "max_tokens": MAX_TOKENS_LONG
        })

        
    combined_items = dataset1_items + dataset2_items

    if not combined_items:
        print("No work items available. Exiting.")
        queue_run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queue_run_task
        return

    seq_items = dataset1_items + dataset2_items
    seq_index = 0
    interleave_toggle = 0

    def next_item() -> Dict[str, Any]:
        nonlocal seq_index, interleave_toggle
        if ORDER_MODE == "shuffle":
            return random.choice(combined_items)
        if ORDER_MODE == "interleave":
            for _ in range(2):
                if interleave_toggle % 2 == 0 and dataset1_items:
                    interleave_toggle += 1
                    return random.choice(dataset1_items)
                if interleave_toggle % 2 == 1 and dataset2_items:
                    interleave_toggle += 1
                    return random.choice(dataset2_items)
                interleave_toggle += 1
            return random.choice(combined_items)

        if not seq_items:
            return random.choice(combined_items)
        item = seq_items[seq_index % len(seq_items)]
        seq_index += 1
        return item

    start_time = time.time()
    end_time = start_time + RUN_SECONDS
    print(f"!!! BENCHMARK START TIME: {start_time} !!!")
    print(f"Pushing for {RUN_SECONDS}s (ORDER_MODE={ORDER_MODE})")
    if push_interval_s > 0:
        print(f"  push_interval_s={push_interval_s:.6f}")
    else:
        print("  push_interval_s=0 (no throttling)")
    print(f"  dataset1: max_tokens={MAX_TOKENS_SHORT}, prefix=ON")
    print(f"  dataset2: max_tokens={MAX_TOKENS_LONG},  prefix=ON")

    pushed_count = 0
    while time.time() < end_time:
        wi = next_item()
        slo = sample_slo()
        final_prompt = wi["final_prompt"]
        max_toks = wi["max_tokens"]

        # 안전: 최종 프롬프트가 컨텍스트를 넘지 않는지 한 번 더 체크(느리면 주석 가능)
        # (여기서는 max_prompt_tokens_{short/long}을 넘으면 skip)
        if wi["dataset"] == "dataset1":
            if token_len(tokenizer, final_prompt, trunc_max=max_prompt_tokens_short) > max_prompt_tokens_short:
                continue
        else:
            if token_len(tokenizer, final_prompt, trunc_max=max_prompt_tokens_long) > max_prompt_tokens_long:
                continue

        q.push(
            prompt=final_prompt,
            model=model_name,
            insertion_time=time.time(),
            slo=slo,
            max_tokens=max_toks,   # <-- execution time 분리 핵심 (패치 필요)
        )
        pushed_count += 1
        if len(pushed_log) < PUSH_LOG_LIMIT:
            pushed_log.append({
                "dataset": wi["dataset"],
                "slo": slo,
                "base_tok_len": wi["base_tok_len"],
                "max_tokens": max_toks,
                "final_prompt_preview": final_prompt[:200].replace("\n", "\\n"),
            })
        if push_interval_s > 0:
            await asyncio.sleep(push_interval_s)

    print(f"Successfully pushed {pushed_count} requests (logged {len(pushed_log)}).")
    if DRAIN_TIMEOUT_S > 0:
        print("Now waiting for processing...")
    # -----------------------------
    # 완료 대기
    # -----------------------------
    try:
        if DRAIN_TIMEOUT_S > 0:
            drain_deadline = time.time() + DRAIN_TIMEOUT_S
            while True:
                total_backpressure = sum(worker.get_backpressure() for worker in q.workers)
                total_queued_groups = sum(len(vq.groups) for vq in q.vq_engine.vqs)
                print(f"Status: [Worker Load: {total_backpressure}] / [Queue Waiting Groups: {total_queued_groups}]")

                if total_backpressure == 0 and total_queued_groups == 0:
                    print("All requests processed. Exiting.")
                    break
                if time.time() >= drain_deadline:
                    print("Drain timeout reached. Exiting.")
                    break

                await asyncio.sleep(5)
    finally:
        queue_run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queue_run_task


if __name__ == "__main__":
    asyncio.run(basic_test())

