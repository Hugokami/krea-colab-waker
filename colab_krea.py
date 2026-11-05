"""
krea-comfy-colab remote engine. Runs inside the persistent Colab kernel via `colab exec -f`.
Engines : moss  (MOSS-Image SFT BF16 + Qwen 3.5 9B INT8 ConvRot + Flux 2 Full VAE + 8-Step DMD Turbo LoRA, Google Drive cached, default)
          krea2 (Krea 2 Turbo INT8 ConvRot + Qwen 3 VL 4B BF16 + Qwen Image VAE, Google Drive cached)
          idol  (Idol-SDK 600M 3D-DiT: Single-image to 3D Gaussian Splats .ply reconstruction)
Modes   : gen | edit | 3d | ui | probe | prefetch                   (env KREA_MODE)
Inputs  : KREA_* env vars; free text is Base64 UTF-8 (KREA_PROMPT_B64 / KREA_NEG_B64)
Output  : /content/krea_out.png or /content/krea_out.ply
"""
import builtins
import os
import re
import sys
import json
import time
import uuid

_orig_print = builtins.print
builtins.print = lambda *a, **k: _orig_print(*a, **dict(k, flush=True))

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
OUT_3D_PATH = "/content/krea_out.ply"
EDIT_INPUT_NAME = "krea_edit_input.png"
CLOUDFLARED = "/content/cloudflared"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_LOG = "/content/cloudflared.log"
TUNNEL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")
IDLE_TIMEOUT_SECONDS = 1500  # 25 minutes
RENDER_TIMEOUT_SECONDS = 1500
MODES = ("gen", "edit", "3d", "ui", "probe", "prefetch", "agent")
ENV_KEYS = ("KREA_MODE", "KREA_ENGINE", "KREA_PROMPT_B64", "KREA_NEG_B64", "KREA_W", "KREA_H",
            "KREA_STEPS", "KREA_CFG", "KREA_SEED", "KREA_INPUT", "KREA_STRENGTH", "KREA_DENOISE",
            "KREA_FAST", "KREA_LORA", "KREA_RES")

MOSS_DMD_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
MOSS_DMD_REMOTE = "lora.safetensors"
MOSS_DMD_LORA = "moss_image_sft_8step_dmd.safetensors"
IDOL_REPO = "https://github.com/speridlabs/idol-sdk.git"
IDOL_DIR = "/content/idol-sdk"

# High-precision production engines cached permanently on Google Drive (>1 TB storage)
ENGINES = {
    "moss": {
        "repo": "Comfy-Org/MOSS-Image",
        "unet": "moss_image_sft_bf16.safetensors",
        "clip": "qwen3.5_9b_int8_convrot.safetensors",
        "vae": "flux2-vae.safetensors",
        "edit_lora": "moss_image_edit_lora_rank_64_bf16.safetensors",
        "control_lora": "moss_image_color_depth_control.safetensors",
        "dmd_lora": MOSS_DMD_LORA,
        "clip_type": "moss_image",
        "steps": 35, "cfg": 4.0, "sampler": "euler", "scheduler": "simple",
        "drive_cache": True,
    },
    "krea2": {
        "repo": "Comfy-Org/Krea-2",
        "unet": "krea2_turbo_int8_convrot.safetensors",
        "clip": "qwen3vl_4b_int8_convrot.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "edit_lora": "krea2_style_reference.safetensors",
        "style_lora": "krea2_style_reference.safetensors",
        "turbo_lora": "krea2_turbo_lora_rank_64_bf16.safetensors",
        "clip_type": "qwen_image",
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "drive_cache": True,
    },
}

MODEL_DIRS = {
    "unet": "diffusion_models",
    "clip": "text_encoders",
    "vae": "vae",
    "edit_lora": "loras",
    "control_lora": "loras",
    "style_lora": "loras",
    "turbo_lora": "loras",
    "dmd_lora": "loras",
    "geometry": "geometry_estimation",
}

RESOLUTIONS = {
    "1mp": {
        "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
        "4:3": (1152, 896), "3:4": (896, 1152), "3:2": (1216, 832), "2:3": (832, 1216),
    },
    "1.5mp": {
        "1:1": (1280, 1280), "16:9": (1600, 896), "9:16": (896, 1600),
        "4:3": (1408, 1056), "3:4": (1056, 1408), "3:2": (1504, 1008), "2:3": (1008, 1504),
    },
    "2k": {
        "1:1": (1440, 1440), "16:9": (1920, 1088), "9:16": (1088, 1920),
        "4:3": (1664, 1248), "3:4": (1248, 1664), "3:2": (1760, 1168), "2:3": (1168, 1760),
    },
}

STYLE_LORAS = {
    "darkbrush": ("krea2_darkbrush.safetensors", "monochrome ink wash style"),
    "dotmatrix": ("krea2_dotmatrix.safetensors", "monochrome stippling style"),
    "kidsdrawing": ("krea2_kidsdrawing.safetensors", "naive expressive sketch style"),
    "neondrip": ("krea2_neondrip.safetensors", "textured abstract style"),
    "rainywindow": ("krea2_rainywindow.safetensors", "rainy window style"),
    "retroanime": ("krea2_retroanime.safetensors", "purple retro anime style"),
    "softwatercolor": ("krea2_softwatercolor.safetensors", "art deco watercolor style"),
    "sunsetblur": ("krea2_sunsetblur.safetensors", "ethereal motion blur style"),
    "vintagetarot": ("krea2_vintagetarot.safetensors", "vintage tarot style"),
}


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
                seed: int, edit_image: str | None = None, lora_strength: float = 1.0, denoise: float = 1.0,
                lora_name: str | None = None, dmd_lora: str | None = None, weight_dtype: str = "fp8_e4m3fn") -> dict:
    """ComfyUI API-format workflow. Node ids 1-9 are fixed; test_krea.py asserts on them.
    fp8_e4m3fn keeps UNet (~8.0GB) resident in 15GB T4 VRAM alongside text encoder, eliminating PCIe bus thrashing.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": eng["unet"], "weight_dtype": weight_dtype}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": eng["clip"], "type": eng["clip_type"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": eng["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model, pos, neg, latent = ["1", 0], ["4", 0], ["5", 0], ["6", 0]

    # Optional style LoRA
    if lora_name and not edit_image:
        g["15"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": lora_name, "strength_model": lora_strength}}
        model = ["15", 0]

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

    # 8-step DMD Turbo LoRA acceleration (3.5x speedup)
    if dmd_lora:
        g["16"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": dmd_lora, "strength_model": 1.0}}
        model = ["16", 0]

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
def mount_drive() -> bool:
    if not os.path.isdir(DRIVE_ROOT):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"[krea] Drive not mounted ({e}); using Colab SSD only.")
    if os.path.isdir(DRIVE_ROOT):
        os.makedirs(DRIVE_CACHE, exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/krea_comfy_outputs", exist_ok=True)
        print(f"[krea] Google Drive mounted and cache active at: {DRIVE_CACHE}")
        return True
    return False


def bootstrap(use_drive: bool = False):
    if os.path.exists(READY_FLAG):
        return
    comfy_archive = f"{DRIVE_CACHE}/comfyui_repo.tar.gz"
    if not os.path.isdir(COMFY_DIR):
        if use_drive and os.path.isfile(comfy_archive) and os.path.getsize(comfy_archive) > 1_000_000:
            print(f"[krea] Restoring ComfyUI directly from Google Drive ({comfy_archive}) ...")
            subprocess.run(["tar", "-xzf", comfy_archive, "-C", "/content"], check=True)
        else:
            print("[krea] First run: cloning ComfyUI and caching to Google Drive ...")
            subprocess.run(["git", "clone", "--depth", "1", COMFY_REPO, COMFY_DIR], check=True)
            if use_drive:
                try:
                    tmp_tar = comfy_archive + ".part"
                    subprocess.run(["tar", "-czf", tmp_tar, "-C", "/content", "ComfyUI"], check=True)
                    os.replace(tmp_tar, comfy_archive)
                    print(f"[krea] Cached ComfyUI repo to Google Drive: {comfy_archive}")
                except Exception as e:
                    print(f"[krea] Notice: could not archive ComfyUI to Drive ({e})")
    print("[krea] Verifying Python dependencies ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY_DIR}/requirements.txt",
                    "huggingface_hub"], check=True)
    open(READY_FLAG, "w").close()


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


def ensure_models(engine: str, mode: str, use_drive: bool, progress_cb=None):
    """SSD target present -> done. Else fast Google Drive cache hit -> else HF download (+ cache to Drive)."""
    eng = ENGINES[engine]
    index = None
    for key, optional in required_files(mode):
        name = eng[key]
        target = f"{COMFY_DIR}/models/{MODEL_DIRS[key]}/{name}"
        if _size(target) > 0:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        cached = f"{DRIVE_CACHE}/{name}"
        if use_drive and _size(cached) > 10_000_000 and not os.path.exists(cached + ".part"):
            print(f"[krea] Google Drive cache hit: {name} ({_size(cached) / 1e9:.2f} GB) -> staging to SSD...")
            if progress_cb:
                try:
                    progress_cb({"phase": f"Staging {name} from Drive...", "step": 0, "total": 1, "elapsed_s": 1, "eta_s": 15})
                except Exception:
                    pass
            _atomic_copy(cached, target)
            _reset_activity()
            continue
        if index is None:
            index = repo_index(eng["repo"])
        repo_path = resolve_repo_path(index, name)
        if repo_path is None:
            msg = f"{name} not in {eng['repo']}. Candidates: {sorted(p for p in index if p.endswith('.safetensors'))}"
            if optional:
                print(f"[krea] Optional file skipped: {msg}")
                continue
            raise RuntimeError(msg)
        if use_drive and _size(cached) == index[repo_path]:
            print(f"[krea] Google Drive cache hit: {name}")
            if progress_cb:
                try:
                    progress_cb({"phase": f"Staging {name} from Drive...", "step": 0, "total": 1, "elapsed_s": 1, "eta_s": 15})
                except Exception:
                    pass
            _atomic_copy(cached, target)
            continue
        sz_gb = index[repo_path] / 1e9
        print(f"[krea] Downloading {repo_path} ({sz_gb:.2f} GB) ...")
        if progress_cb:
            try:
                progress_cb({"phase": f"Downloading {name} ({sz_gb:.1f} GB)...", "step": 0, "total": 1, "elapsed_s": 1, "eta_s": int(sz_gb * 20)})
            except Exception:
                pass
        from huggingface_hub import hf_hub_download
        os.replace(hf_hub_download(eng["repo"], repo_path, local_dir=HF_DL_DIR), target)
        if use_drive:
            try:
                if progress_cb:
                    try:
                        progress_cb({"phase": f"Caching {name} ({sz_gb:.1f} GB) to Drive...", "step": 0, "total": 1, "elapsed_s": 1, "eta_s": 25})
                    except Exception:
                        pass
                print(f"[krea] Saving {name} permanently to Google Drive ({cached}) ...")
                _atomic_copy(target, cached)
                print(f"[krea] Cached to Google Drive: {cached}")
            except OSError as e:
                print(f"[krea] Drive cache write failed ({e}); continuing from SSD.")
        _reset_activity()


def ensure_lora(repo: str, lora_name: str, use_drive: bool, remote_name: str | None = None, progress_cb=None):
    """Download / stage a single LoRA model directly to models/loras."""
    target = f"{COMFY_DIR}/models/loras/{lora_name}"
    if _size(target) > 0:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    cached = f"{DRIVE_CACHE}/{lora_name}"
    if use_drive and _size(cached) > 10_000_000 and not os.path.exists(cached + ".part"):
        print(f"[krea] Google Drive cache hit: {lora_name} -> staging to SSD...")
        _atomic_copy(cached, target)
        return
    index = repo_index(repo)
    lookup = remote_name or lora_name
    repo_path = resolve_repo_path(index, lookup)
    if not repo_path:
        print(f"[krea] Warning: LoRA {lookup} not found in {repo}")
        return
    print(f"[krea] Downloading LoRA {lora_name} from {repo}/{repo_path} ({index[repo_path] / 1e6:.1f} MB) ...")
    if progress_cb:
        try:
            progress_cb({"phase": f"Downloading LoRA {lora_name}...", "step": 0, "total": 1, "elapsed_s": 1, "eta_s": 10})
        except Exception:
            pass
    from huggingface_hub import hf_hub_download
    dl = hf_hub_download(repo, repo_path, local_dir=HF_DL_DIR)
    _atomic_copy(dl, target)
    if use_drive:
        try:
            _atomic_copy(target, cached)
            print(f"[krea] Cached {lora_name} to Google Drive: {cached}")
        except Exception as e:
            print(f"[krea] Drive cache notice: {e}")


# ---------------------------------------------------------------------------
# Idol-SDK 3D Object Reconstruction (.ply Gaussian Splats)
# ---------------------------------------------------------------------------
def bootstrap_idol(use_drive: bool = False):
    if os.path.isdir(IDOL_DIR):
        return
    print("[krea] Installing idol-sdk for 3D Gaussian Splat reconstruction...")
    subprocess.run(["git", "clone", "--depth", "1", IDOL_REPO, IDOL_DIR], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=IDOL_DIR, check=True)


def run_idol_3d(input_image: str, prompt: str = "real", use_drive: bool = False) -> str:
    bootstrap_idol(use_drive)
    try:
        _http("/free", {"unload_models": True, "free_memory": True}, timeout=10)
    except Exception:
        pass
    out_dir = "/content/idol_out"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    mode_param = "anime" if "anime" in prompt.lower() else "real"
    cmd = [sys.executable, "-m", "idol.sample", "--image", input_image, "--mode", mode_param, "--out", out_dir]
    print(f"[krea-3d] Running Idol-SDK: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=IDOL_DIR, check=True)
    ply_files = [os.path.join(r, f) for r, _, fs in os.walk(out_dir) if f.endswith(".ply")]
    if not ply_files:
        raise RuntimeError("Idol-SDK finished without producing a .ply file.")
    _atomic_copy(ply_files[0], OUT_3D_PATH)
    return OUT_3D_PATH


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
    print("[krea] Starting ComfyUI server with normal VRAM management for 16GB T4 GPU ...")
    log = open(COMFY_LOG, "ab")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--dont-print-server"],
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


def _parse_mmss(s: str) -> int:
    try:
        parts = [int(p) for p in s.strip().split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        pass
    return 0


def _parse_comfy_progress(t0: float, default_total: int = 35) -> dict | None:
    try:
        if not os.path.isfile(COMFY_LOG):
            return None
        with open(COMFY_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - 2048), os.SEEK_SET)
            chunk = f.read().decode("utf-8", errors="replace")
        segments = [seg.strip() for seg in re.split(r"[\r\n]+", chunk) if seg.strip()]
        for seg in reversed(segments):
            m = re.search(r"(\d+)/(\d+)\s+\[([0-9:]+)<([0-9:?]+)", seg)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                el_s = _parse_mmss(m.group(3)) or int(time.time() - t0)
                eta_raw = m.group(4)
                if "?" in eta_raw or step == 0:
                    eta_s = max(5, int((total - step) * 3.5))
                else:
                    eta_s = _parse_mmss(eta_raw)
                phase = "Loading models into T4 GPU VRAM..." if step == 0 else "Denoising on T4 GPU"
                return {"phase": phase, "step": step, "total": total, "elapsed_s": el_s, "eta_s": eta_s}
            if "Requested to load" in seg or "Model Initializing" in seg:
                el_s = int(time.time() - t0)
                return {"phase": "Loading models into T4 GPU VRAM...", "step": 0, "total": default_total, "elapsed_s": el_s, "eta_s": max(15, 115 - el_s)}
    except Exception:
        pass
    return None


def run_prompt(graph: dict, timeout: int = RENDER_TIMEOUT_SECONDS, progress_cb=None) -> str:
    try:
        prompt_id = _http("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=30)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI rejected workflow: {e.read().decode('utf-8', 'replace')[:3000]}") from None
    t0 = time.time()
    last_prog_time = 0.0
    last_prog_key = None
    deadline = t0 + timeout
    while time.time() < deadline:
        _reset_activity()
        now = time.time()
        if progress_cb and (now - last_prog_time >= 4.0):
            prog = _parse_comfy_progress(t0)
            if prog:
                pkey = (prog.get("phase"), prog.get("step"))
                if pkey != last_prog_key or (now - last_prog_time >= 10.0):
                    last_prog_key = pkey
                    last_prog_time = now
                    try:
                        progress_cb(prog)
                    except Exception:
                        pass
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


def render(engine: str, mode: str, env: dict, use_drive: bool = False):
    if mode == "3d":
        print(f"[krea] Generating 3D object from {env.get('KREA_INPUT')} via Idol-SDK...")
        out_ply = run_idol_3d(env["KREA_INPUT"], b64d(env["KREA_PROMPT_B64"]), use_drive)
        print(f"KREA_COMPLETE:{out_ply}")
        return

    eng = ENGINES[engine]
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    prompt = b64d(env["KREA_PROMPT_B64"]).strip()
    if not prompt:
        raise ValueError("Prompt/instruction cannot be empty.")
    
    is_fast = bool(env.get("KREA_FAST") in ("1", "true", "True"))
    dmd_lora = None

    if engine == "moss":
        if is_fast:
            ensure_lora(MOSS_DMD_REPO, MOSS_DMD_LORA, use_drive, remote_name=MOSS_DMD_REMOTE)
            dmd_lora = MOSS_DMD_LORA
            steps = int(env["KREA_STEPS"] or 8)
            cfg = float(env["KREA_CFG"] or 1.0)
        else:
            steps = int(env["KREA_STEPS"] or 35)
            cfg = float(env["KREA_CFG"] or 4.0)
    else:
        steps = int(env["KREA_STEPS"] or (8 if is_fast else 12))
        cfg = float(env["KREA_CFG"] or 1.0)

    seed = int(env["KREA_SEED"]) if env["KREA_SEED"] else random.randint(0, 2**32 - 1)
    edit_image = None
    if mode == "edit":
        width, height = prepare_edit_input(env["KREA_INPUT"])
        edit_image = EDIT_INPUT_NAME
    else:
        res_tier = env.get("KREA_RES", "1mp")
        res_dict = RESOLUTIONS.get(res_tier, RESOLUTIONS["1mp"])
        ar = f"{env.get('KREA_W', '1024')}:{env.get('KREA_H', '1024')}"
        width = int(env["KREA_W"] or 1024)
        height = int(env["KREA_H"] or 1024)

    lora_name = env.get("KREA_LORA") or None
    if lora_name and lora_name in STYLE_LORAS:
        lora_file, trigger = STYLE_LORAS[lora_name]
        ensure_lora("Comfy-Org/Krea-2", lora_file, use_drive)
        lora_name = lora_file
        if trigger.lower() not in prompt.lower():
            prompt = f"{trigger}, {prompt}"

    graph = build_graph(eng, prompt, b64d(env["KREA_NEG_B64"]), width, height, steps, cfg, seed,
                        edit_image=edit_image, lora_strength=float(env["KREA_STRENGTH"] or 1.0),
                        denoise=float(env["KREA_DENOISE"] or 1.0), lora_name=lora_name, dmd_lora=dmd_lora)
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
            if key in eng:
                path = resolve_repo_path(index, eng[key])
                missing |= path is None
                print(f"  [{name}] {key:<9} {'OK ' + path if path else 'MISSING ' + eng[key]}")
        if missing:
            print(f"  [{name}] repo .safetensors: {sorted(p for p in index if p.endswith('.safetensors'))}")
    print("KREA_PROBE_DONE")


def _start_heartbeat_thread(worker_url: str, headers: dict):
    def _hb():
        while True:
            try:
                req = urllib.request.Request(f"{worker_url}/agent/next?ping=1", data=b"{}", headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as r:
                    pass
            except Exception:
                pass
            time.sleep(15)
    t = threading.Thread(target=_hb, daemon=True)
    t.start()


def _prewarm_vram(worker_url: str, headers: dict):
    """If no job is already waiting in D1 queue, run a 1-step 256x256 pass so UNet, CLIP, DMD LoRA, and VAE are hot in T4 VRAM."""
    try:
        eng = ENGINES["moss"]
        print("[krea-agent] Pre-warming T4 GPU VRAM (loading FP8 UNet + Qwen CLIP + 8-step DMD LoRA)...")
        warm_graph = build_graph(eng, "warmup", "", 256, 256, 1, 1.0, 42, dmd_lora=MOSS_DMD_LORA, weight_dtype="fp8_e4m3fn")
        run_prompt(warm_graph, timeout=180)
        print("[krea-agent] T4 GPU VRAM pre-warmed! First user prompt will render immediately with 0s model load delay.")
    except Exception as e:
        print(f"[krea-agent] Pre-warm notice (non-fatal): {e}")


def agent_main():
    worker_url = os.environ.get("KREA_WORKER_URL", "").rstrip("/")
    agent_secret = os.environ.get("KREA_AGENT_SECRET", "")
    if not worker_url or not agent_secret:
        raise ValueError("KREA_WORKER_URL and KREA_AGENT_SECRET must be set for agent mode.")

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers = {
        "Authorization": f"Bearer {agent_secret}",
        "Content-Type": "application/json",
        "User-Agent": ua,
    }

    print(f"[krea-agent] Starting Colab Agent for Worker: {worker_url}")
    # Immediate heartbeat so Telegram shows ONLINE within 2s of container start
    _start_heartbeat_thread(worker_url, headers)

    use_drive = mount_drive()
    bootstrap(use_drive)

    # Start ComfyUI server in parallel while staging models from Google Drive to SSD
    comfy_thread = threading.Thread(target=start_comfy, daemon=True)
    comfy_thread.start()

    print("[krea-agent] Staging MOSS-Image gen models + 8-step DMD Turbo LoRA from Google Drive...")
    ensure_models("moss", "gen", use_drive)
    ensure_lora(MOSS_DMD_REPO, MOSS_DMD_LORA, use_drive, remote_name=MOSS_DMD_REMOTE)
    comfy_thread.join()
    start_comfy()  # verify alive

    prewarmed = False
    print("[krea-agent] ComfyUI ready. Polling for jobs from Cloudflare D1...")

    last_activity = time.time()
    last_hb = 0.0
    last_err = 0.0

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
            if time.time() - last_hb > 60:
                print(f"[krea-agent] Heartbeat active ({time.strftime('%H:%M:%S')}) - listening for Telegram jobs...")
                last_hb = time.time()
        except Exception as e:
            if time.time() - last_err > 15:
                print(f"[krea-agent] Poll error: {e}")
                last_err = time.time()
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
            if not prewarmed:
                prewarmed = True
                _prewarm_vram(worker_url, headers)
                last_activity = time.time()
            continue
        prewarmed = True

        last_activity = time.time()
        jid = job["id"]
        mode = job["mode"]
        engine = job.get("engine", "moss")
        ar = job.get("ar", "1:1")
        res_tier = job.get("res", "1mp")
        quality_mode = job.get("quality", "fast")
        lora_mode = job.get("lora", "none")
        prompt = job["prompt"]

        print(f"\n[krea-agent] Claimed job {jid}: {engine} {mode} (AR={ar}, Res={res_tier}, Quality={quality_mode}, LoRA={lora_mode})\nPrompt: {prompt}")

        try:
            def send_progress(prog: dict):
                nonlocal last_activity
                last_activity = time.time()
                _reset_activity()
                try:
                    p_url = f"{worker_url}/agent/progress/{jid}"
                    p_req = urllib.request.Request(p_url, data=json.dumps(prog).encode("utf-8"), headers=headers, method="POST")
                    urllib.request.urlopen(p_req, timeout=8)
                except Exception:
                    pass

            # Handle 3D object reconstruction via Idol-SDK
            if mode == "3d":
                input_url = f"{worker_url}/agent/input/{jid}"
                in_req = urllib.request.Request(input_url, headers={
                    "Authorization": f"Bearer {agent_secret}",
                    "User-Agent": ua,
                })
                local_in = f"{COMFY_DIR}/input/idol_in_{jid}.png"
                os.makedirs(f"{COMFY_DIR}/input", exist_ok=True)
                with urllib.request.urlopen(in_req, timeout=60) as r, open(local_in, "wb") as f:
                    f.write(r.read())

                send_progress({"phase": "Reconstructing 3D Gaussian Splats (Idol-SDK)...", "step": 1, "total": 2, "elapsed_s": 1, "eta_s": 12})
                ply_path = run_idol_3d(local_in, prompt, use_drive)
                send_progress({"phase": "Uploading 3D .ply model to Telegram...", "step": 2, "total": 2, "elapsed_s": int(time.time() - last_activity), "eta_s": 2})

                with open(ply_path, "rb") as f:
                    ply_bytes = f.read()

                render_s = max(1, int(time.time() - last_activity))
                done_url = f"{worker_url}/agent/done/{jid}?render_s={render_s}"
                done_req = urllib.request.Request(done_url, data=ply_bytes, headers={
                    "Authorization": f"Bearer {agent_secret}",
                    "Content-Type": "application/octet-stream",
                    "User-Agent": ua,
                }, method="POST")
                with urllib.request.urlopen(done_req, timeout=60) as r:
                    print(f"[krea-agent] 3D Job {jid} delivered to Telegram successfully!")
                last_activity = time.time()
                continue

            # 2D Image Generation & Editing
            ensure_models(engine, mode, use_drive, progress_cb=send_progress)
            eng = ENGINES[engine]
            dmd_lora = None

            # Determine steps & CFG by quality preset
            if engine == "moss":
                if quality_mode == "fast":
                    ensure_lora(MOSS_DMD_REPO, MOSS_DMD_LORA, use_drive, remote_name=MOSS_DMD_REMOTE, progress_cb=send_progress)
                    dmd_lora = MOSS_DMD_LORA
                    steps = 8
                    cfg = 1.0
                    est_time = 28
                else:
                    steps = 35
                    cfg = 4.0
                    est_time = 115
            else:
                steps = 8 if quality_mode == "fast" else 12
                cfg = 1.0
                est_time = 25

            seed = random.randint(0, 2**32 - 1)
            edit_image = None
            lora_name = None

            # Handle style LoRA
            if lora_mode != "none" and lora_mode in STYLE_LORAS:
                lora_file, trigger = STYLE_LORAS[lora_mode]
                ensure_lora("Comfy-Org/Krea-2", lora_file, use_drive, progress_cb=send_progress)
                lora_name = lora_file
                if trigger.lower() not in prompt.lower():
                    prompt = f"{trigger}, {prompt}"

            if mode == "edit":
                input_url = f"{worker_url}/agent/input/{jid}"
                in_req = urllib.request.Request(input_url, headers={
                    "Authorization": f"Bearer {agent_secret}",
                    "User-Agent": ua,
                })
                local_in = f"{COMFY_DIR}/input/{EDIT_INPUT_NAME}"
                os.makedirs(f"{COMFY_DIR}/input", exist_ok=True)
                with urllib.request.urlopen(in_req, timeout=60) as r, open(local_in, "wb") as f:
                    f.write(r.read())
                width, height = prepare_edit_input(local_in)
                edit_image = EDIT_INPUT_NAME
            else:
                res_dict = RESOLUTIONS.get(res_tier, RESOLUTIONS["1mp"])
                width, height = res_dict.get(ar, (1024, 1024))

            send_progress({"phase": f"Starting {engine.upper()} on T4 GPU...", "step": 0, "total": steps, "elapsed_s": 1, "eta_s": est_time})

            graph = build_graph(eng, prompt, "", width, height, steps, cfg, seed,
                                edit_image=edit_image, lora_name=lora_name, dmd_lora=dmd_lora)
            out_file = run_prompt(graph, progress_cb=send_progress)
            send_progress({"phase": "Decoding full VAE & uploading PNG...", "step": steps, "total": steps, "elapsed_s": int(time.time() - last_activity), "eta_s": 3})

            with open(out_file, "rb") as f:
                img_bytes = f.read()

            if use_drive:
                try:
                    drive_out = f"{DRIVE_ROOT}/krea_comfy_outputs/krea_job{jid}_{int(time.time())}.png"
                    with open(drive_out, "wb") as df:
                        df.write(img_bytes)
                    print(f"[krea-agent] Saved backup to Google Drive: {drive_out}")
                except Exception as de:
                    print(f"[krea-agent] Drive output backup notice: {de}")

            render_s = max(1, int(time.time() - last_activity))
            done_url = f"{worker_url}/agent/done/{jid}?render_s={render_s}"
            done_req = urllib.request.Request(done_url, data=img_bytes, headers={
                "Authorization": f"Bearer {agent_secret}",
                "Content-Type": "image/png",
                "User-Agent": ua,
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
            try:
                start_comfy()
            except Exception:
                pass
            last_activity = time.time()
            last_activity = time.time()


def main():
    env = {k: os.environ.pop(k, "") for k in ENV_KEYS}
    mode = env["KREA_MODE"] or "gen"
    engine = env["KREA_ENGINE"] or "moss"
    if mode == "agent":
        agent_main()
        return
    _init_watchdog()
    _reset_activity()
    if mode not in MODES or (mode != "3d" and engine not in ENGINES):
        raise ValueError(f"Bad KREA_MODE={mode!r} or KREA_ENGINE={engine!r}")
    use_drive = (mode == "3d" or ENGINES[engine]["drive_cache"]) and mount_drive()
    bootstrap(use_drive)
    if mode == "probe":
        probe()
        return
    if mode != "3d":
        ensure_models(engine, mode, use_drive)
    start_comfy()
    if mode == "prefetch":
        print(f"KREA_READY:{engine}")
    elif mode == "ui":
        print(f"KREA_UI_URL:{start_tunnel()}")
    else:
        render(engine, mode, env, use_drive=use_drive)
    _reset_activity()


if __name__ == "__main__":
    main()
