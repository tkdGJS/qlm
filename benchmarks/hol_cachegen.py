# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from qlm.queue.queue import Queue
from qlm.endpoints.endpoint import Endpoint

import asyncio
import contextlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Protocol, cast

from transformers import AutoTokenizer


_DEFAULT_LMCACHE_RESULT_DIR = "/home/noslab-gpu/tkdgjs/qlm/result"


def _round_up_int(x: int, base: int) -> int:
    if base <= 0:
        return x
    r = x % base
    return x if r == 0 else (x + (base - r))


async def _wait_for_vllm_ready(address: str, port: int, timeout_s: float) -> None:
    import requests

    deadline = time.time() + max(1.0, timeout_s)
    url_models = f"http://{address}:{port}/v1/models"
    url_metrics = f"http://{address}:{port}/metrics"
    last_err: str | None = None

    while time.time() < deadline:
        try:
            r = requests.get(url_models, timeout=2.0)
            if r.status_code == 200 and '"object":"list"' in (r.text or ""):
                m = requests.get(url_metrics, timeout=2.0)
                if m.status_code == 200 and (m.text or ""):
                    return
        except Exception as e:
            last_err = str(e)
        await asyncio.sleep(1.0)

    raise RuntimeError(f"vLLM not ready within {timeout_s}s (last_err={last_err})")


def _get_vram_gb() -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.used / (1024**3)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return (total_b - free_b) / (1024**3)
    except Exception:
        pass
    return 0.0


def _setup_vram_log_env() -> str:
    _env_default("LMCACHE_RESULT_DIR", _DEFAULT_LMCACHE_RESULT_DIR)
    base = (os.environ.get("LMCACHE_VRAM_LOG_FILE", "") or "").strip()
    if not base:
        base = os.path.join(os.environ["LMCACHE_RESULT_DIR"], "lmcache_vram.log")

    base_dir = os.path.dirname(base)
    base_name = os.path.basename(base)
    if base_name.startswith("multi_req_"):
        multi = base
    else:
        multi = os.path.join(base_dir, f"multi_req_{base_name}")

    os.environ["LMCACHE_VRAM_LOG"] = "1"
    os.environ["LMCACHE_VRAM_LOG_FILE"] = multi
    os.makedirs(os.path.dirname(multi) or ".", exist_ok=True)
    try:
        with open(multi, "w", encoding="utf-8"):
            pass
    except Exception:
        pass
    return multi


def _append_vram_log_line(
    prefix: str,
    abs_ts_s: float,
    event: str,
    vram_gb: float,
    metadata: dict[str, object] | None = None,
) -> None:
    result_dir = os.environ.get("LMCACHE_RESULT_DIR", "./result")
    log_file = (os.environ.get("LMCACHE_VRAM_LOG_FILE", "") or "").strip()
    if not log_file:
        log_file = os.path.join(result_dir, "lmcache_vram.log")

    meta_str = ""
    if metadata:
        meta_str = " ".join(f"{k}={v}" for k, v in metadata.items())

    line = f"{prefix} {abs_ts_s:.6f} {event} {vram_gb:.4f}GB"
    if meta_str:
        line += " " + meta_str
    line += "\n"

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        _ = f.write(line)
        f.flush()


def _parse_ts_to_seconds(ts_str: str) -> float | None:
    try:
        return float(ts_str)
    except Exception:
        pass
    try:
        s = ts_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        return None


def _parse_metadata_kv(rest: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for tok in (rest or "").strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        out[k] = v
    return out


def _extract_unified_vram_log(log_path: str, config: dict[str, object]) -> str | None:
    scheduler_events_by_request_id: dict[str, dict[str, float | None]] = {}
    lmcache_events: list[dict[str, object]] = []
    client_events: list[dict[str, object]] = []

    sched_re = re.compile(
        r"^\[VLLM_SCHEDULER\]\s+(\S+)\s+(request_start|prefill_complete|request_finish)\b"
    )
    sched_rid_re = re.compile(r"\brequest_id\s*[=:]\s*([^\s,]+)")

    lmcache_re = re.compile(r"^\[LMCACHE_VRAM\](?:\[(?P<sub>[^\]]+)\])?\s+(?P<ts>\S+)\s*(?P<rest>.*)$")
    client_re = re.compile(r"^\[(?P<prefix>CLIENT_REQUEST|CLIENT_SAMPLER)\]\s+(?P<ts>\S+)\s+(?P<event>\S+)\b(?P<rest>.*)$")
    vram_re = re.compile(r"^\s*(?P<vram>[0-9.]+)GB\b\s*(?P<rest>.*)$")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip("\n")
                if not line:
                    continue

                if "[VLLM_SCHEDULER]" in line:
                    m = sched_re.search(line)
                    if not m:
                        continue
                    ts = _parse_ts_to_seconds(m.group(1))
                    if ts is None:
                        continue
                    event = m.group(2)
                    rid_m = sched_rid_re.search(line)
                    if not rid_m:
                        continue
                    rid = rid_m.group(1).strip()
                    if not rid:
                        continue

                    cur = scheduler_events_by_request_id.get(
                        rid,
                        {
                            "request_start_ts": None,
                            "prefill_complete_ts": None,
                            "request_finish_ts": None,
                        },
                    )
                    if event == "request_start":
                        cur["request_start_ts"] = ts
                    elif event == "prefill_complete":
                        cur["prefill_complete_ts"] = ts
                    elif event == "request_finish":
                        cur["request_finish_ts"] = ts
                    scheduler_events_by_request_id[rid] = cur
                    continue

                if line.startswith("[LMCACHE_VRAM]") or line.startswith("[LMCACHE_VRAM]["):
                    m = lmcache_re.match(line)
                    if not m:
                        continue
                    ts = _parse_ts_to_seconds(m.group("ts") or "")
                    if ts is None:
                        continue
                    rest = (m.group("rest") or "").strip()
                    event = rest
                    lm_rec: dict[str, object] = {"ts": ts, "event": event}
                    sub = (m.group("sub") or "").strip()
                    if sub:
                        lm_rec["subprefix"] = sub
                    lmcache_events.append(lm_rec)
                    continue

                if "[CLIENT_REQUEST]" in line or "[CLIENT_SAMPLER]" in line:
                    m = client_re.match(line)
                    if not m:
                        continue
                    ts = _parse_ts_to_seconds(m.group("ts") or "")
                    if ts is None:
                        continue

                    event = (m.group("event") or "").strip()
                    rest = m.group("rest") or ""

                    vram_gb: float | None = None
                    meta_str = rest
                    vram_m = vram_re.match(rest)
                    if vram_m:
                        try:
                            vram_gb = float(vram_m.group("vram"))
                        except Exception:
                            vram_gb = None
                        meta_str = vram_m.group("rest") or ""

                    metadata = _parse_metadata_kv(meta_str)
                    client_events.append(
                        {
                            "ts": ts,
                            "event": event,
                            "vram_gb": vram_gb,
                            "metadata": metadata,
                        }
                    )
                    continue
    except Exception:
        return None

    scheduler_out: dict[str, dict[str, float | None]] = {}
    for rid, sched_rec in scheduler_events_by_request_id.items():
        rs = sched_rec.get("request_start_ts")
        pc = sched_rec.get("prefill_complete_ts")
        rf = sched_rec.get("request_finish_ts")
        ttft_s = (pc - rs) if (isinstance(rs, float) and isinstance(pc, float)) else None
        ttlt_s = (rf - rs) if (isinstance(rs, float) and isinstance(rf, float)) else None
        scheduler_out[rid] = {
            "request_start_ts": rs,
            "prefill_complete_ts": pc,
            "request_finish_ts": rf,
            "ttft_s": ttft_s,
            "ttlt_s": ttlt_s,
        }

    result_dir = os.environ.get("LMCACHE_RESULT_DIR", _DEFAULT_LMCACHE_RESULT_DIR)
    out_dir = os.path.dirname(log_path) or result_dir
    out_path = os.path.join(out_dir, f"extract_{os.path.basename(log_path)}.json")

    payload = {
        "config": config,
        "scheduler_events_by_request_id": scheduler_out,
        "lmcache_events": lmcache_events,
        "client_events": client_events,
    }
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        return None
    return out_path


async def _client_sampler(
    stop: asyncio.Event,
    interval_s: float,
    mode: str,
    max_tokens: int,
    slo_s: float,
    slo_type: int,
) -> None:
    while not stop.is_set():
        try:
            _append_vram_log_line(
                "[CLIENT_SAMPLER]",
                time.time(),
                "periodic_sample",
                _get_vram_gb(),
                {
                    "mode": mode,
                    "max_tokens": max_tokens,
                    "slo": f"{slo_s:.6f}",
                    "slo_type": slo_type,
                },
            )
        except Exception:
            pass

        try:
            _ = await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            continue


def _setup_running_req_log_file(vram_log_file: str) -> str:
    base_dir = os.path.dirname(vram_log_file) or os.getcwd()
    base_name = os.path.basename(vram_log_file)
    if base_name.startswith("multi_req_"):
        base_name = base_name[len("multi_req_") :]
    out = os.path.join(base_dir, f"running_req_{base_name}")
    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8"):
            pass
    except Exception:
        pass
    return out


def _read_vllm_metric(text: str, metric_name: str) -> float | None:
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return float(parts[-1])
        except Exception:
            return None
    return None


async def _vllm_running_req_sampler(
    stop: asyncio.Event,
    interval_s: float,
    address: str,
    port: int,
    out_path: str,
) -> None:
    import httpx

    url = f"http://{address}:{port}/metrics"
    timeout = httpx.Timeout(connect=0.5, read=1.0, write=1.0, pool=0.5)
    last_running = 0
    last_waiting = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        while not stop.is_set():
            ts = time.time()
            running = last_running
            waiting = last_waiting
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    text = r.text or ""
                    rv = _read_vllm_metric(text, "vllm:num_requests_running")
                    wv = _read_vllm_metric(text, "vllm:num_requests_waiting")
                    if rv is not None:
                        running = int(rv)
                    if wv is not None:
                        waiting = int(wv)
                    last_running = running
                    last_waiting = waiting
            except Exception:
                pass

            try:
                with open(out_path, "a", encoding="utf-8") as f:
                    _ = f.write(f"[RUNNING_REQ] {ts:.6f} {running} {waiting}\n")
                    f.flush()
            except Exception:
                pass

            try:
                _ = await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                continue


class _Tokenizer(Protocol):
    def encode(self, text: str, **kwargs: object) -> list[int]: ...


def token_len(tokenizer: _Tokenizer, text: str, trunc_max: int | None = None) -> int:
    if trunc_max is None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=trunc_max + 1,
    )
    return len(ids)


def _print_task_exception(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc:
        print("run_queue exception:", exc)


def _usage() -> None:
    p = os.path.basename(sys.argv[0] or "hol_cachegen.py")
    print(f"Usage: python {p} native|cachegen")


def _select_lmcache_config(mode: str) -> str:
    if mode == "native":
        return "/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml"
    if mode == "cachegen":
        return "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml"
    raise ValueError(f"unknown mode: {mode}")


def _env_default(key: str, default: str) -> None:
    if key not in os.environ or os.environ[key] == "":
        os.environ[key] = default
    return None


def _clear_lmcache_disk(config_file: str) -> None:
    """
    Read LMCACHE_CONFIG_FILE and clear the local_disk directory before
    starting vLLM, ensuring each run starts from a clean state.

    Why this matters:
    - lmcache_cachegen.yaml (local_cpu=true): serves hits from CPU memory;
      only 4 serialize events (disk persistence), deserialize=0.
    - lmcache_native_diskread.yaml (local_cpu=false): serves every hit from
      disk; 4784 deserialize events per 600s run.
    - Without clearing, a leftover disk cache skips the serialize phase and
      immediately starts deserializing from the previous run's data.
    """
    import glob
    import shutil

    if not config_file:
        return

    local_disk_path: str | None = None
    try:
        with open(config_file, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("local_disk:"):
                    val = stripped[len("local_disk:"):].strip().strip('"').strip("'")
                    if val.startswith("file://"):
                        val = val[len("file://"):]
                    local_disk_path = val.rstrip("/")
                    break
    except Exception as e:
        print(f"[clear_lmcache_disk] WARNING: could not read config {config_file}: {e}")
        return

    if not local_disk_path:
        print("[clear_lmcache_disk] no local_disk entry found, skipping.")
        return

    if not os.path.isdir(local_disk_path):
        print(f"[clear_lmcache_disk] disk path does not exist yet: {local_disk_path}")
        return

    files = glob.glob(os.path.join(local_disk_path, "*"))
    if not files:
        print(f"[clear_lmcache_disk] disk already empty: {local_disk_path}")
        return

    print(f"[clear_lmcache_disk] clearing {len(files)} file(s) from {local_disk_path}")
    for fp in files:
        try:
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception as e:
            print(f"[clear_lmcache_disk] WARNING: failed to remove {fp}: {e}")
    print("[clear_lmcache_disk] done.")


def _load_tokenizer(model_name: str) -> _Tokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


async def basic_test() -> None:
    if len(sys.argv) != 2:
        _usage()
        raise SystemExit(2)

    mode = (sys.argv[1] or "").strip().lower()
    if mode not in ("native", "cachegen"):
        _usage()
        raise SystemExit(2)

    DATASET_PATH = os.environ.get("DATASET_PATH", "data/ShareGPT_V3_unfiltered_cleaned_split.json")
    RUN_TIME_S = float(os.environ.get("RUN_TIME_S", "600"))
    TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "100"))
    PUSH_INTERVAL_S = float(os.environ.get("PUSH_INTERVAL_S", "0.1"))
    DRAIN_TIMEOUT_S = float(os.environ.get("DRAIN_TIMEOUT_S", "0"))
    MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "8192"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "0"))  # 0 = no limit (vLLM model default)
    SLO_S = float(os.environ.get("SLO_S", "1.0"))
    _env_default("CLIENT_SAMPLE_INTERVAL_S", "0.1")
    CLIENT_SAMPLE_INTERVAL_S = float(os.environ.get("CLIENT_SAMPLE_INTERVAL_S", "0.1"))
    _env_default("RUNNING_REQ_SAMPLE_INTERVAL_S", "0.1")
    RUNNING_REQ_SAMPLE_INTERVAL_S = float(os.environ.get("RUNNING_REQ_SAMPLE_INTERVAL_S", "0.1"))
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")

    desired_max_len = _round_up_int(MAX_INPUT_TOKENS + MAX_TOKENS, 256)
    _env_default("VLLM_MAX_MODEL_LEN", str(desired_max_len))
    _env_default("VLLM_MAX_NUM_SEQS", "128")
    _env_default("VLLM_MAX_NUM_BATCHED_TOKENS", "131072")
    _env_default("VLLM_GPU_MEMORY_UTILIZATION", "0.7")
    _env_default("VLLM_DTYPE", "half")
    _env_default("VLLM_ENABLE_CHUNKED_PREFILL", "1")

    # Allow caller to override config path without editing this script.
    if not (os.environ.get("LMCACHE_CONFIG_FILE", "") or "").strip():
        os.environ["LMCACHE_CONFIG_FILE"] = _select_lmcache_config(mode)

    vram_log_file = _setup_vram_log_env()

    print("=== HOL CacheGen Benchmark ===")
    print(
        "cfg:",
        f"mode={mode}",
        f"LMCACHE_CONFIG_FILE={os.environ['LMCACHE_CONFIG_FILE']}",
        f"LMCACHE_VRAM_LOG_FILE={vram_log_file}",
        f"RUN_TIME_S={RUN_TIME_S}",
        f"TARGET_COUNT={TARGET_COUNT}",
        f"PUSH_INTERVAL_S={PUSH_INTERVAL_S}",
        f"MAX_TOKENS={MAX_TOKENS}",
        f"SLO_S={SLO_S}",
        "SLO_RULE=(<=128?1:200)",
        f"DRAIN_TIMEOUT_S={DRAIN_TIMEOUT_S}",
        f"CLIENT_SAMPLE_INTERVAL_S={CLIENT_SAMPLE_INTERVAL_S}",
        f"RUNNING_REQ_SAMPLE_INTERVAL_S={RUNNING_REQ_SAMPLE_INTERVAL_S}",
    )
    print(
        "vllm:",
        f"VLLM_MAX_MODEL_LEN={os.environ['VLLM_MAX_MODEL_LEN']}",
        f"VLLM_MAX_NUM_SEQS={os.environ['VLLM_MAX_NUM_SEQS']}",
        f"VLLM_MAX_NUM_BATCHED_TOKENS={os.environ['VLLM_MAX_NUM_BATCHED_TOKENS']}",
        f"VLLM_GPU_MEMORY_UTILIZATION={os.environ['VLLM_GPU_MEMORY_UTILIZATION']}",
        f"VLLM_DTYPE={os.environ['VLLM_DTYPE']}",
        f"VLLM_ENABLE_CHUNKED_PREFILL={os.environ.get('VLLM_ENABLE_CHUNKED_PREFILL','')}",
    )

    random.seed(42)
    tokenizer = _load_tokenizer(model_name)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        raw_list = cast(list[object], json.load(f))

    prompt_min_tokens_s = (os.environ.get("PROMPT_MIN_TOKENS", "") or "").strip()
    try:
        prompt_min_tokens = int(prompt_min_tokens_s) if prompt_min_tokens_s else 0
    except Exception:
        prompt_min_tokens = 0

    prompt_pool_limit_s = (os.environ.get("PROMPT_POOL_LIMIT", "") or "").strip()
    try:
        prompt_pool_limit = int(prompt_pool_limit_s) if prompt_pool_limit_s else 0
    except Exception:
        prompt_pool_limit = 0

    prompt_pool: list[str] = []
    for item_obj in raw_list:
        if not isinstance(item_obj, dict):
            continue
        conv_obj = item_obj.get("conversations")
        if not isinstance(conv_obj, list) or len(conv_obj) < 2:
            continue
        a0_obj, b0_obj = conv_obj[0], conv_obj[1]
        if not isinstance(a0_obj, dict) or not isinstance(b0_obj, dict):
            continue
        if a0_obj.get("from") != "human" or b0_obj.get("from") != "gpt":
            continue
        p_obj = a0_obj.get("value")
        if not isinstance(p_obj, str):
            continue
        p = p_obj.strip()
        if not p:
            continue
        tlen = token_len(tokenizer, p, trunc_max=MAX_INPUT_TOKENS)
        if tlen > MAX_INPUT_TOKENS:
            continue
        if prompt_min_tokens > 0 and tlen < prompt_min_tokens:
            continue
        prompt_pool.append(p)

        if prompt_pool_limit > 0 and len(prompt_pool) >= prompt_pool_limit:
            break

    print(f"Prompt pool size (<= {MAX_INPUT_TOKENS} input toks): {len(prompt_pool)}")
    if not prompt_pool:
        raise RuntimeError("No prompts available after filtering")

    address = os.environ.get("VLLM_ADDRESS", "localhost")
    port = int(os.environ.get("VLLM_PORT", "8000"))
    vllm_startup_wait_s = float(os.environ.get("VLLM_STARTUP_WAIT_S", "10"))
    vllm_ready_timeout_s = float(os.environ.get("VLLM_READY_TIMEOUT_S", "180"))
    qlm_warmup_s = float(os.environ.get("QLM_WARMUP_S", "5"))
    gpu_index = int(os.environ.get("GPU_INDEX", "0"))

    endpoint: Endpoint | None = None
    q: Queue | None = None
    queue_run_task: asyncio.Task[object] | None = None
    sampler_stop: asyncio.Event | None = None
    sampler_task: asyncio.Task[object] | None = None
    running_req_log_file: str | None = None
    running_req_task: asyncio.Task[object] | None = None

    _clear_lmcache_disk(os.environ.get("LMCACHE_CONFIG_FILE", ""))
    try:
        endpoint = Endpoint(address=address, port=port, model=model_name)
        print("Waiting for vLLM server to start...")
        await asyncio.sleep(vllm_startup_wait_s)
        await _wait_for_vllm_ready(address, port, timeout_s=vllm_ready_timeout_s)

        q = Queue()
        q.register_worker(
            address,
            port,
            endpoint,
            max_model_len=int(os.environ["VLLM_MAX_MODEL_LEN"]),
            max_num_seqs=int(os.environ["VLLM_MAX_NUM_SEQS"]),
            max_num_batched_tokens=int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]),
            gpu_index=gpu_index,
        )

        queue_run_task = asyncio.create_task(q.run_queue())
        queue_run_task.add_done_callback(_print_task_exception)
        await asyncio.sleep(qlm_warmup_s)

        sampler_stop = asyncio.Event()
        sampler_task = asyncio.create_task(
            _client_sampler(
                sampler_stop,
                CLIENT_SAMPLE_INTERVAL_S,
                mode,
                MAX_TOKENS,
                SLO_S,
                0,
            )
        )
        sampler_task.add_done_callback(_print_task_exception)

        running_req_log_file = _setup_running_req_log_file(vram_log_file)
        running_req_task = asyncio.create_task(
            _vllm_running_req_sampler(
                sampler_stop,
                RUNNING_REQ_SAMPLE_INTERVAL_S,
                address,
                port,
                running_req_log_file,
            )
        )
        running_req_task.add_done_callback(_print_task_exception)

        pushed_count = 0
        t0 = time.monotonic()
        next_due = t0
        if RUN_TIME_S > 0:
            deadline = t0 + RUN_TIME_S
            i = 0
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                if PUSH_INTERVAL_S > 0:
                    if next_due > now:
                        await asyncio.sleep(next_due - now)
                    else:
                        next_due = now
                    next_due += PUSH_INTERVAL_S
                else:
                    await asyncio.sleep(0)

                prompt = random.choice(prompt_pool)

                slo_in_toks = token_len(tokenizer, prompt, trunc_max=128)
                if slo_in_toks <= 128:
                    slo_s, slo_type = 1.0, 0
                else:
                    slo_s, slo_type = 200.0, 1

                in_toks = token_len(tokenizer, prompt, trunc_max=MAX_INPUT_TOKENS)

                try:
                    _append_vram_log_line(
                        "[CLIENT_REQUEST]",
                        time.time(),
                        "request_start",
                        _get_vram_gb(),
                        {
                            "mode": mode,
                            "idx": i,
                            "in_toks": in_toks,
                            "max_tokens": MAX_TOKENS,
                            "slo": f"{slo_s:.6f}",
                            "slo_type": slo_type,
                        },
                    )
                except Exception:
                    pass

                q.push(
                    prompt=prompt,
                    model=model_name,
                    insertion_time=time.time(),
                    slo=slo_s,
                    max_tokens=MAX_TOKENS if MAX_TOKENS > 0 else None,
                    slo_type=slo_type,
                )
                pushed_count += 1
                i += 1
        else:
            for i in range(TARGET_COUNT):
                now = time.monotonic()
                if PUSH_INTERVAL_S > 0:
                    if next_due > now:
                        await asyncio.sleep(next_due - now)
                    else:
                        next_due = now
                    next_due += PUSH_INTERVAL_S
                else:
                    await asyncio.sleep(0)

                prompt = random.choice(prompt_pool)

                slo_in_toks = token_len(tokenizer, prompt, trunc_max=128)
                if slo_in_toks <= 128:
                    slo_s, slo_type = 1.0, 0
                else:
                    slo_s, slo_type = 200.0, 1

                in_toks = token_len(tokenizer, prompt, trunc_max=MAX_INPUT_TOKENS)

                try:
                    _append_vram_log_line(
                        "[CLIENT_REQUEST]",
                        time.time(),
                        "request_start",
                        _get_vram_gb(),
                        {
                            "mode": mode,
                            "idx": i,
                            "in_toks": in_toks,
                            "max_tokens": MAX_TOKENS,
                            "slo": f"{slo_s:.6f}",
                            "slo_type": slo_type,
                        },
                    )
                except Exception:
                    pass

                q.push(
                    prompt=prompt,
                    model=model_name,
                    insertion_time=time.time(),
                    slo=slo_s,
                    max_tokens=MAX_TOKENS if MAX_TOKENS > 0 else None,
                    slo_type=slo_type,
                )
                pushed_count += 1

        print(
            f"Pushed {pushed_count} requests (max_tokens={MAX_TOKENS}, interval={PUSH_INTERVAL_S}, run_time_s={RUN_TIME_S})"
        )

        if DRAIN_TIMEOUT_S > 0 and q is not None:
            drain_deadline = time.time() + DRAIN_TIMEOUT_S
            while True:
                try:
                    total_backpressure = sum(worker.get_backpressure() for worker in q.workers)
                except Exception:
                    total_backpressure = 0
                total_queued_groups = sum(len(vq.groups) for vq in q.vq_engine.vqs)
                print(
                    "Status:",
                    f"worker_backpressure={total_backpressure}",
                    f"queued_groups={total_queued_groups}",
                )
                if total_backpressure == 0 and total_queued_groups == 0:
                    break
                if time.time() >= drain_deadline:
                    break
                await asyncio.sleep(5)

        cfg: dict[str, object] = {
            "mode": mode,
            "model_name": model_name,
            "target_count": TARGET_COUNT,
            "run_time_s": RUN_TIME_S,
            "pushed_count": pushed_count,
            "push_interval_s": PUSH_INTERVAL_S,
            "max_tokens": MAX_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "vram_log_file": vram_log_file,
            "running_req_log_file": running_req_log_file,
        }
        out_path = _extract_unified_vram_log(vram_log_file, cfg)
        if out_path:
            print(f"Wrote log extract: {out_path}")
    finally:
        if sampler_stop is not None:
            sampler_stop.set()
        if sampler_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await sampler_task
        if running_req_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await running_req_task
        if queue_run_task is not None:
            _ = queue_run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await queue_run_task
        if endpoint is not None:
            with contextlib.suppress(Exception):
                endpoint._stop_vllm_server()


if __name__ == "__main__":
    asyncio.run(basic_test())
