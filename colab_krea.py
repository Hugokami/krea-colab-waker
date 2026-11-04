"""
krea-comfy-colab remote engine. Runs inside the persistent Colab kernel via `colab exec -f`.
Engines : moss  (MOSS-Image SFT INT8 + official edit LoRA, Google Drive cached, default)
          krea2 (Krea 2 Turbo FP8, Colab SSD only: ~18 GB exceeds 15 GB free Drive)
Modes   : gen | edit | ui | probe | prefetch                       (env KREA_MODE)
Inputs  : KREA_* env vars; free text is Base64 UTF-8 (KREA_PROMPT_B64 / KREA_NEG_B64)
Output  : /content/krea_out.png + stdout markers KREA_COMPLETE: / KREA_UI_URL: / KREA_READY:
All side effects live in main(); importing this module locally is safe (test_krea.py).
"""
import os
import re
import sys
import json
import time
import uuid
import base64
import random
import shutil
import threading
import subprocess
import urllib.error
import urllib.request

COMFY_DIR = "/content/ComfyUI"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_LOG = "/content/comfy.log"
READY_FLAG = "/content/.krea_ready"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_CACHE = f"{DRIVE_ROOT}/comfy_cache"
HF_DL_DIR = "/content/hf_dl"
OUT_PATH = "/content/krea_out.png"
EDIT_INPUT_NAME = "krea_edit_input.png"
CLOUDFLARED = "/content/cloudflared"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_LOG = "/content/cloudflared.log"
TUNNEL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")
IDLE_TIMEOUT_SECONDS = 1500  # 25 minutes
RENDER_TIMEOUT_SECONDS = 1500
MODES = ("gen", "edit", "ui", "probe", "prefetch", "agent")
ENV_KEYS = ("KREA_MODE", "KREA_ENGINE", "KREA_PROMPT_B64", "KREA_NEG_B64", "KREA_W", "KREA_H",
            "KREA_STEPS", "KREA_CFG", "KREA_SEED", "KREA_INPUT", "KREA_STRENGTH", "KREA_DENOISE")

# Filenames from concepts_brief.md. `krea-colab probe` checks them against the live HF repo.
ENGINES = {
    "moss": {
        "repo": "Comfy-Org/MOSS-Image",
        "unet": "moss_image_sft_int8_convrot.safetensors",
        "clip": "qwen3.5_9b_w4a8_gptq.safetensors",
        "vae": "flux2-vae_small.safetensors",
        "edit_lora": "moss_image_edit_lora_rank_64_bf16.safetensors",
        "clip_type": "moss_image",
        "steps": 28, "cfg": 4.0, "sampler": "euler", "scheduler": "simple",
        "drive_cache": True,
    },
    "krea2": {
        "repo": "Comfy-Org/Krea-2",
        "unet": "krea2_turbo_fp8_scaled.safetensors",
        "clip": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "edit_lora": "krea2_identity_edit.safetensors",
        "clip_type": "qwen_image",
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "drive_cache": False,
    },
}
MODEL_DIRS = {"unet": "diffusion_models", "clip": "text_encoders", "vae": "vae", "edit_lora": "loras"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline by test_krea.py)
# ---------------------------------------------------------------------------
def b64d(value: str) -> str:
    return base64.b64decode(value.encode("ascii")).decode("utf-8") if value else ""


def combo_options(spec) -> list:
    """object_info combo spec: legacy [[opts], {...}] or new ["COMBO", {"options": [...]}]."""
    if spec and isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return []


def resolve_repo_path(index: dict, basename: str) -> str | None:
    """Match by basename so both `diffusion_models/x` and `split_files/diffusion_models/x` layouts work."""
    hits = sorted(p for p in index if p.rsplit("/", 1)[-1] == basename)
    return hits[0] if hits else None


def required_files(mode: str) -> list[tuple[str, bool]]:
    """(ENGINES key, optional) pairs needed by a mode."""
    files = [("unet", False), ("clip", False), ("vae", False)]
    if mode == "edit":
        files.append(("edit_lora", False))
    elif mode in ("ui", "prefetch"):
        files.append(("edit_lora", True))
    return files


def edit_dims(w: int, h: int, max_pixels: int = 1024 * 1024, multiple: int = 16) -> tuple[int, int]:
    """Downscale to <= max_pixels (never upscale), round down to multiples of 16."""
    scale = min(1.0, (max_pixels / float(w * h)) ** 0.5)
    return (max(multiple, int(w * scale) // multiple * multiple),
            max(multiple, int(h * scale) // multiple * multiple))


def build_graph(eng: dict, prompt: str, negative: str, width: int, height: int, steps: int, cfg: float,
                seed: int, edit_image: str | None = None, lora_strength: float = 1.0, denoise: float = 1.0) -> dict:
    """ComfyUI API-format workflow. Node ids are fixed; test_krea.py asserts on them."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": eng["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": eng["clip"], "type": eng["clip_type"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": eng["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model, pos, neg, latent = ["1", 0], ["4", 0], ["5", 0], ["6", 0]
    if edit_image:
        g["10"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": eng["edit_lora"], "strength_model": lora_strength}}
        g["11"] = {"class_type": "LoadImage", "inputs": {"image": edit_image}}
        g["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
        g["13"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": ["12", 0]}}
        g["14"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": ["12", 0]}}
        model, pos, neg = ["10", 0], ["13", 0], ["14", 0]
        if denoise < 1.0:
            latent = ["12", 0]  # img2img start from source image
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": eng["sampler"], "scheduler": eng["scheduler"],
        "positive": pos, "negative": neg, "latent_image": latent, "denoise": denoise}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "krea"}}
    return g


def validate_graph(graph: dict, info: dict, clip_type: str) -> list[str]:
    errors = []
    missing = sorted({n["class_type"] for n in graph.values()} - set(info))
    if missing:
        errors.append(f"ComfyUI lacks nodes {missing}; this ComfyUI build does not support the engine.")
    if "CLIPLoader" in info:
        opts = combo_options(info["CLIPLoader"]["input"]["required"]["type"])
        if clip_type not in opts:
            errors.append(f"CLIPLoader type '{clip_type}' unsupported; available: {opts}")
    return errors


# ---------------------------------------------------------------------------
# Idle watchdog (25 min, ComfyUI-queue aware)
# ---------------------------------------------------------------------------
def _reset_activity():
    globals()["_LAST_ACTIVITY"] = time.time()


def _http(path: str, payload=None, timeout: float = 10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _queue_busy() -> bool:
    try:
        q = _http("/queue", timeout=3)
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:
        return False


def _idle_watchdog():
    while True:
        time.sleep(30)
        if _queue_busy():
            _reset_activity()
            continue
        idle = time.time() - globals()["_LAST_ACTIVITY"]
        if idle > IDLE_TIMEOUT_SECONDS:
            print(f"\n[krea] Idle {round(idle / 60, 1)} min. Unassigning runtime to save compute units...")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                print(f"[krea] Auto-unassign notice: {e}")
                globals()["_WATCHDOG_RUNNING"] = False
            break


def _init_watchdog():
    globals().setdefault("_LAST_ACTIVITY", time.time())
    if not globals().get("_WATCHDOG_RUNNING"):
        globals()["_WATCHDOG_RUNNING"] = True
        threading.Thread(target=_idle_watchdog, daemon=True).start()
        print("[krea] Idle watchdog armed (25 min, ComfyUI-queue aware).")


# ---------------------------------------------------------------------------
# Bootstrap, Drive, model cache
# ---------------------------------------------------------------------------
def bootstrap():
    if os.path.exists(READY_FLAG):
        return
    print("[krea] First run on this VM: installing latest ComfyUI ...")
    if not os.path.isdir(COMFY_DIR):
        subprocess.run(["git", "clone", "--depth", "1", COMFY_REPO, COMFY_DIR], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY_DIR}/requirements.txt",
                    "huggingface_hub"], check=True)
    open(READY_FLAG, "w").close()


def mount_drive() -> bool:
    if not os.path.isdir(DRIVE_ROOT):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"[krea] Drive not mounted ({e}); using Colab SSD only.")
    if os.path.isdir(DRIVE_ROOT):
        os.makedirs(DRIVE_CACHE, exist_ok=True)
        return True
    return False


def repo_index(repo: str) -> dict:
    from huggingface_hub import HfApi
    return {f.path: f.size for f in HfApi().list_repo_tree(repo, recursive=True)
            if getattr(f, "size", None) is not None}


def _size(path: str) -> int:
    return os.path.getsize(path) if os.path.isfile(path) else -1


def _atomic_copy(src: str, dst: str):
    tmp = dst + ".part"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def ensure_models(engine: str, mode: str, use_drive: bool):
    """SSD target present -> done. Else Drive cache (size-verified vs HF) -> else HF download (+ cache to Drive)."""
    eng = ENGINES[engine]
    index = None
    for key, optional in required_files(mode):
        name = eng[key]
        target = f"{COMFY_DIR}/models/{MODEL_DIRS[key]}/{name}"
        if _size(target) > 0:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if index is None:
            index = repo_index(eng["repo"])
        repo_path = resolve_repo_path(index, name)
        if repo_path is None:
            msg = f"{name} not in {eng['repo']}. Candidates: {sorted(p for p in index if p.endswith('.safetensors'))}"
            if optional:
                print(f"[krea] Optional file skipped: {msg}")
                continue
            raise RuntimeError(msg)
        cached = f"{DRIVE_CACHE}/{name}"
        if use_drive and _size(cached) == index[repo_path]:
            print(f"[krea] Drive cache hit: {name}")
            _atomic_copy(cached, target)
            continue
        print(f"[krea] Downloading {repo_path} ({index[repo_path] / 1e9:.2f} GB) ...")
        from huggingface_hub import hf_hub_download
        os.replace(hf_hub_download(eng["repo"], repo_path, local_dir=HF_DL_DIR), target)
        if use_drive:
            try:
                _atomic_copy(target, cached)
                print(f"[krea] Cached to Drive: {cached}")
            except OSError as e:
                print(f"[krea] Drive cache write failed ({e}); continuing from SSD.")
        _reset_activity()


# ---------------------------------------------------------------------------
# ComfyUI server, rendering, tunnel
# ---------------------------------------------------------------------------
def comfy_alive() -> bool:
    try:
        _http("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _tail(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def start_comfy():
    if comfy_alive():
        return
    print("[krea] Starting ComfyUI server ...")
    log = open(COMFY_LOG, "ab")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                            cwd=COMFY_DIR, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(240):
        if comfy_alive():
            print("[krea] ComfyUI ready.")
            return
        if proc.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError(f"ComfyUI failed to start. Tail of {COMFY_LOG}:\n{_tail(COMFY_LOG)}")


def object_info() -> dict:
    return _http("/object_info", timeout=60)


def run_prompt(graph: dict, timeout: int = RENDER_TIMEOUT_SECONDS) -> str:
    try:
        prompt_id = _http("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=30)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI rejected workflow: {e.read().decode('utf-8', 'replace')[:3000]}") from None
    deadline = time.time() + timeout
    while time.time() < deadline:
        _reset_activity()
        try:
            hist = _http(f"/history/{prompt_id}", timeout=10)
        except (urllib.error.URLError, OSError):
            if not comfy_alive():
                raise RuntimeError(f"ComfyUI crashed (OOM?). Tail of {COMFY_LOG}:\n{_tail(COMFY_LOG)}")
            hist = {}
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"Workflow failed: {json.dumps(status.get('messages', []))[-3000:]}")
            images = entry.get("outputs", {}).get("9", {}).get("images", [])
            if images:
                img = images[0]
                return os.path.join(COMFY_DIR, "output", img.get("subfolder", ""), img["filename"])
            if status.get("completed"):
                raise RuntimeError("Workflow completed without images.")
        time.sleep(1)
    raise RuntimeError(f"Render timed out after {timeout}s.")


def prepare_edit_input(src: str) -> tuple[int, int]:
    if not src or not os.path.isfile(src):
        raise FileNotFoundError(f"Edit input not uploaded: {src!r}")
    from PIL import Image, ImageOps
    with Image.open(src) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    w, h = edit_dims(*im.size)
    os.makedirs(f"{COMFY_DIR}/input", exist_ok=True)
    im.resize((w, h), Image.LANCZOS).save(f"{COMFY_DIR}/input/{EDIT_INPUT_NAME}")
    return w, h


def render(engine: str, mode: str, env: dict):
    eng = ENGINES[engine]
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    prompt = b64d(env["KREA_PROMPT_B64"]).strip()
    if not prompt:
        raise ValueError("Prompt/instruction cannot be empty.")
    steps = int(env["KREA_STEPS"] or eng["steps"])
    cfg = float(env["KREA_CFG"] or eng["cfg"])
    seed = int(env["KREA_SEED"]) if env["KREA_SEED"] else random.randint(0, 2**32 - 1)
    edit_image = None
    if mode == "edit":
        width, height = prepare_edit_input(env["KREA_INPUT"])
        edit_image = EDIT_INPUT_NAME
    else:
        width, height = int(env["KREA_W"] or 1024), int(env["KREA_H"] or 1024)
    graph = build_graph(eng, prompt, b64d(env["KREA_NEG_B64"]), width, height, steps, cfg, seed,
                        edit_image=edit_image, lora_strength=float(env["KREA_STRENGTH"] or 1.0),
                        denoise=float(env["KREA_DENOISE"] or 1.0))
    errors = validate_graph(graph, object_info(), eng["clip_type"])
    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"[krea] {engine} {mode} {width}x{height} steps={steps} cfg={cfg} seed={seed}")
    t0 = time.time()
    _atomic_copy(run_prompt(graph), OUT_PATH)
    print(f"[krea] Done in {time.time() - t0:.1f}s -> {OUT_PATH}")
    print(f"KREA_COMPLETE:{OUT_PATH} seed={seed}")


def start_tunnel() -> str:
    prev = globals().get("_TUNNEL")
    if prev and prev[0].poll() is None:
        return prev[1]
    if not os.path.isfile(CLOUDFLARED):
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED + ".part")
        os.chmod(CLOUDFLARED + ".part", 0o755)
        os.replace(CLOUDFLARED + ".part", CLOUDFLARED)
    log = open(CLOUDFLARED_LOG, "wb")
    proc = subprocess.Popen([CLOUDFLARED, "tunnel", "--url", COMFY_URL, "--no-autoupdate"],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        time.sleep(1)
        m = TUNNEL_RE.search(_tail(CLOUDFLARED_LOG, 200))
        if m:
            globals()["_TUNNEL"] = (proc, m.group(0))
            return m.group(0)
        if proc.poll() is not None:
            break
    proc.kill()
    raise RuntimeError(f"cloudflared produced no URL. Log:\n{_tail(CLOUDFLARED_LOG)}")


def probe():
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"])
    with open("/proc/meminfo") as f:
        mem = {ln.split(":")[0]: int(ln.split()[1]) // 1024 for ln in f if ln.startswith(("MemTotal", "MemAvailable"))}
    print(f"[krea] RAM MB {mem} | /content free {shutil.disk_usage('/content').free / 1e9:.0f} GB"
          f" | Drive mounted: {mount_drive()}")
    if mem.get("MemTotal", 0) < 14000:
        print("[krea] NOTE: <14 GB system RAM; krea2 (~13 GB diffusion file) may be OOM-killed while loading.")
    start_comfy()
    info = object_info()
    for name, eng in ENGINES.items():
        graph = build_graph(eng, "probe", "", 1024, 1024, 1, 1.0, 0, edit_image=EDIT_INPUT_NAME)
        for err in validate_graph(graph, info, eng["clip_type"]):
            print(f"  [{name}] NODES: {err}")
        try:
            index = repo_index(eng["repo"])
        except Exception as e:
            print(f"  [{name}] REPO ERROR {eng['repo']}: {e}")
            continue
        missing = False
        for key in MODEL_DIRS:
            path = resolve_repo_path(index, eng[key])
            missing |= path is None
            print(f"  [{name}] {key:<9} {'OK ' + path if path else 'MISSING ' + eng[key]}")
        if missing:
            print(f"  [{name}] repo .safetensors: {sorted(p for p in index if p.endswith('.safetensors'))}")
    print("KREA_PROBE_DONE")


def agent_main():
    worker_url = os.environ.get("KREA_WORKER_URL", "").rstrip("/")
    agent_secret = os.environ.get("KREA_AGENT_SECRET", "")
    if not worker_url or not agent_secret:
        raise ValueError("KREA_WORKER_URL and KREA_AGENT_SECRET must be set for agent mode.")

    print(f"[krea-agent] Starting Colab Agent for Worker: {worker_url}")
    bootstrap()
    use_drive = mount_drive()
    print("[krea-agent] Pre-loading default MOSS-Image models...")
    ensure_models("moss", "gen", use_drive)
    start_comfy()
    print("[krea-agent] ComfyUI ready. Polling for jobs from Cloudflare D1...")

    last_activity = time.time()
    headers = {
        "Authorization": f"Bearer {agent_secret}",
        "Content-Type": "application/json",
    }

    while True:
        time.sleep(3)
        # Check 25-minute idle timeout
        if time.time() - last_activity > IDLE_TIMEOUT_SECONDS:
            print(f"[krea-agent] Idle for {round(IDLE_TIMEOUT_SECONDS / 60)} minutes. Releasing GPU...")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                print(f"[krea-agent] Unassign notice: {e}")
            break

        try:
            req = urllib.request.Request(f"{worker_url}/agent/next", data=b"{}", headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as r:
                res = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue

        if res.get("stop"):
            print("[krea-agent] Stop requested via Telegram. Releasing GPU...")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                print(f"[krea-agent] Unassign notice: {e}")
            break

        job = res.get("job")
        if not job:
            continue

        last_activity = time.time()
        jid = job["id"]
        mode = job["mode"]
        engine = job.get("engine", "moss")
        ar = job.get("ar", "1:1")
        prompt = job["prompt"]
        print(f"\n[krea-agent] Claimed job {jid}: {engine} {mode} ({ar})\nPrompt: {prompt}")

        try:
            ensure_models(engine, mode, use_drive)
            eng = ENGINES[engine]
            steps = eng["steps"]
            cfg = eng["cfg"]
            seed = random.randint(0, 2**32 - 1)
            edit_image = None

            if mode == "edit":
                input_url = f"{worker_url}/agent/input/{jid}"
                in_req = urllib.request.Request(input_url, headers={"Authorization": f"Bearer {agent_secret}"})
                local_in = f"{COMFY_DIR}/input/{EDIT_INPUT_NAME}"
                os.makedirs(f"{COMFY_DIR}/input", exist_ok=True)
                with urllib.request.urlopen(in_req, timeout=60) as r, open(local_in, "wb") as f:
                    f.write(r.read())
                width, height = prepare_edit_input(local_in)
                edit_image = EDIT_INPUT_NAME
            else:
                aspects = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344), "4:3": (1152, 896),
                           "3:4": (896, 1152), "3:2": (1216, 832), "2:3": (832, 1216)}
                width, height = aspects.get(ar, (1024, 1024))

            graph = build_graph(eng, prompt, "", width, height, steps, cfg, seed, edit_image=edit_image)
            out_file = run_prompt(graph)
            with open(out_file, "rb") as f:
                img_bytes = f.read()

            done_url = f"{worker_url}/agent/done/{jid}"
            done_req = urllib.request.Request(done_url, data=img_bytes, headers={
                "Authorization": f"Bearer {agent_secret}",
                "Content-Type": "image/png",
            }, method="POST")
            with urllib.request.urlopen(done_req, timeout=60) as r:
                print(f"[krea-agent] Job {jid} delivered to Telegram successfully!")
            last_activity = time.time()
        except Exception as err:
            print(f"[krea-agent] Job {jid} failed: {err}")
            try:
                fail_url = f"{worker_url}/agent/fail/{jid}"
                fail_req = urllib.request.Request(fail_url, data=json.dumps({"error": str(err)}).encode("utf-8"), headers=headers, method="POST")
                urllib.request.urlopen(fail_req, timeout=15)
            except Exception:
                pass
            last_activity = time.time()


def main():
    _init_watchdog()
    _reset_activity()
    env = {k: os.environ.pop(k, "") for k in ENV_KEYS}
    mode = env["KREA_MODE"] or "gen"
    engine = env["KREA_ENGINE"] or "moss"
    if mode == "agent":
        agent_main()
        return
    if mode not in MODES or engine not in ENGINES:
        raise ValueError(f"Bad KREA_MODE={mode!r} or KREA_ENGINE={engine!r}")
    bootstrap()
    if mode == "probe":
        probe()
        return
    use_drive = ENGINES[engine]["drive_cache"] and mount_drive()
    ensure_models(engine, mode, use_drive)
    start_comfy()
    if mode == "prefetch":
        print(f"KREA_READY:{engine}")
    elif mode == "ui":
        print(f"KREA_UI_URL:{start_tunnel()}")
    else:
        render(engine, mode, env)
    _reset_activity()


if __name__ == "__main__":
    main()
