# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
import io
import itertools
import json
import mimetypes
import os
import queue
import random
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from PIL import Image

ALL_CATEGORIES = [
    "add", "remove", "replace", "extract", "adjust",
    "compose", "motion", "text", "background", "color",
    "beauty", "lowlevel", "stylize", "viewpoint", "portrait"
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# 不修改：沿用原system prompt
CONSTRUCTION_LOGIC = """
You are an expert Vision-Language Data Engineer specializing in creating high-quality, diverse, and complex image editing datasets. 
You are given a <Reference Image>, its <Image Caption>, and an assigned <Target Category>.

### STEP 1: Feasibility Check
First, analyze the image.
- Example: If category is "portrait" or "beauty", but there are no prominent human, it is NOT suitable.
- Example: If category is "motion", but there is no human/animal with clear limbs/body, it is NOT suitable.
- Example: If category is "text", but there is no existing text or obvious flat surface to add text, it is NOT suitable.
- Example: If category is "extract", but there is no salient object, it is NOT suitable.

Please select as many of the following categories as possible for editing instruction: **viewpoint, extract, beauty, portrait, lowlevel, compose, adjust, stylize**.
Please select as many of the following categories as possible for editing instruction: **viewpoint, extract, beauty, portrait, lowlevel, compose, adjust, stylize**.

### STEP 2: Generate Instructions (If Feasible)

**CRITICAL RULES FOR THE EDIT PROMPT:**
1. **Salient & Noticeable:** The edit MUST be visually significant. Do NOT propose microscopic or overly subtle changes (e.g., "change the color of a tiny button in the background" or "add a small pebble") that are hard to perceive at a glance.
2. **Concise & Direct:** The instruction must be an imperative command. Do NOT over-explain, use redundant justifications, or add filler words inside the prompt. Strictly match the crisp, direct, and constraint-focused format of the examples below.

Identify the THREE (3) most suitable and visually significant editing categories for this specific image from the list below. 
For each selected category, generate one precise, imperative editing instruction. Do NOT generate overly simplistic prompts. Learn from the complexity, constraints, and specific details shown in the examples below:

1. "add": Propose adding a logical object. Specify position, state, or dynamic interaction. 
   - Examples: "Add a man running in sportswear on the road", "Add two eggs next to the animal's feet in the picture", "Add a flock of birds perched on high branches, their silhouettes standing out against the sky, with one bird flying in the frame creating a dynamic contrast."
2. "remove": Identify a distinct object or global artifacts to erase, keeping the background intact. 
   - Examples: "Remove the small fish-shaped toy", "Remove the falling snowflakes to keep the image clean and clear."
3. "replace": Swap an existing object/pattern with a new one, keeping specific constraints unchanged. 
   - Examples: "Replace the white coffee cup with a transparent glass cup, keeping the coffee inside unchanged", "Replace the cartoon elk on the cup with a cartoon penguin wearing a Santa hat."
4. "extract": Isolate a specific object/subject on a white background, excluding specific attachments. 
   - Examples: "Extract the schoolbag in the photo, excluding the person carrying it", "Extract the digital clock above the classroom blackboard."
5. "adjust": Alter specific attributes (material, transparency, texture) WITHOUT changing geometry. 
   - Examples: "Make the drinks in all cups clearer and more translucent, while retaining their respective colors", "Adjust the color of the lizard to match the color of the bark/rock it is lying on, creating a camouflage effect."
6. "compose": Propose a complex hybrid edit involving at least TWO DIFFERENT operations on different objects/regions. 
   - Examples: "Change the material of the two candy canes to metal, and replace the wooden tabletop with a water surface", "Change the background to a jungle and remove the smartwatch."
7. "motion": Change the pose, gesture, or action of a person/animal, including interactions. 
   - Examples: "Make the man in the photo stand up and pet the dog's face", "Adjust the subject's sitting pose to legs crossed, naturally placed, upper body straight, one hand resting on the leg, the other hand hanging naturally", "Make the woman in the picture run on the beach."
8. "text": Add, modify, or replace text. MUST include the exact text in quotes and specify formatting/location. 
   - Examples: "Change 'OK' to 'WOW'", "On the billboard, below 'LOVE', add a line of small text in the same font and color: 'we stay, we grow, we heal'."
9. "background": Replace the environment, ensuring lighting and atmospheric harmony with the foreground. 
   - Examples: "Replace the buildings in the background with the Roman Colosseum, keeping the person and street in the foreground unchanged", "Change the background to the Great Wall and ensure the tone and lighting of the new background remain harmonious and unified with the person and dog in the foreground."
10. "color": Change the color of specific objects, preserving textures/details. 
    - Examples: "Change the color of the leaves from green to autumn golden yellow", "Turn the dark gray keycaps on the keyboard to gold, keeping the letters and symbols on the keycaps unchanged."
11. "beauty": For human faces/heads, suggest beautification, hair adjustment, or structural makeup. 
    - Examples: "Remove forehead wrinkles and even out skin tone", "Naturally fill in the sparse parts of the forehead hairline with hair."
12. "lowlevel": Propose image restoration, enhancement, or specific degradation. 
    - Examples: "Restore this old photo by removing scratches, stains, mildew, and damages, while enhancing image clarity and details, keeping the original tone and lighting unchanged", "Help me remove blur and noise from the photo and improve clarity."
13. "stylize": Apply a specific artistic style to the entire image, mentioning brushstrokes or texture details. 
    - Examples: "Convert the image to an 8-bit pixel art style", "Process the image into a Chinese ink painting style, highlighting the brushstrokes of the bamboo and the variations in ink density."
14. "viewpoint": Change the camera angle, focus, or 3D viewpoint of the scene/object. 
    - Examples: "Rotate the camera 45 degrees to the right", "Focus on the cups on the table for a close-up shot", "Rotate the object 180° around its vertical axis to show its opposite side."
15. "portrait": Complex identity-preserving edits. You can change posture, camera angle, accessories, and background simultaneously, BUT explicitly demand that the facial identity remains perfectly unchanged. 
    - Examples: "Focus the lens on the person's upper body, remove the bouquet in hand, adjust the posture to arms crossed over the chest, and replace the background with a blurred natural landscape", "Adjust the person's posture to a sitting position with legs bent and together, hands placed naturally on the ground, and remove the fan. Change to a high-angle perspective, add reflective texture to the ground, and let the skirt hem stack naturally."

### Output Format (Return ONLY a valid JSON list of 3 objects):
[
    {
        "category": "category_name_1",
        "edit_prompt": "Specific instruction...",
        "reasoning": "Why this is a top choice for this image."
    },
    {
        "category": "category_name_2",
        "edit_prompt": "...",
        "reasoning": "..."
    },
    {
        "category": "category_name_3",
        "edit_prompt": "...",
        "reasoning": "..."
    }
]
"""

SUPPORTED_RESOLUTIONS = [
    {"aspect_ratio": "1:1", "width": 1024, "height": 1024},
    {"aspect_ratio": "2:3", "width": 848, "height": 1264},
    {"aspect_ratio": "3:2", "width": 1264, "height": 848},
    {"aspect_ratio": "3:4", "width": 896, "height": 1200},
    {"aspect_ratio": "4:3", "width": 1200, "height": 896},
    {"aspect_ratio": "4:5", "width": 928, "height": 1152},
    {"aspect_ratio": "5:4", "width": 1152, "height": 928},
    {"aspect_ratio": "9:16", "width": 768, "height": 1376},
    {"aspect_ratio": "16:9", "width": 1376, "height": 768},
]


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return default


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_round(meta: Dict[str, Any], turn: int, create: bool = False) -> Optional[Dict[str, Any]]:
    rounds = meta.setdefault("rounds", [])
    for r in rounds:
        if isinstance(r, dict) and int(r.get("turn", -1)) == turn:
            return r
    if create:
        r = {"turn": turn}
        rounds.append(r)
        rounds.sort(key=lambda x: int(x.get("turn", 0)))
        return r
    return None


def parse_prompt_turns(meta: Dict[str, Any]) -> Set[int]:
    turns: Set[int] = set()
    rounds = meta.get("rounds", [])
    if not isinstance(rounds, list):
        return turns
    for r in rounds:
        if not isinstance(r, dict):
            continue
        t = r.get("turn")
        p = r.get("prompt", {})
        if not isinstance(p, dict):
            continue
        if str(p.get("edit_prompt", "")).strip():
            try:
                turns.add(int(t))
            except Exception:
                continue
    return turns


def find_contiguous_max_image_turn(sample_dir: Path, max_turn: int) -> int:
    t = 0
    while t <= max_turn and (sample_dir / f"turn_{t}.jpg").exists():
        t += 1
    return t - 1


def parse_response_text_to_json_list(text: str) -> List[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    match = re.search(r"(\[[\s\S]*\])", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            pass

    match = re.search(r"(\{[\s\S]*\})", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

    return []


def discover_samples(samples_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not samples_root.exists():
        return out
    with os.scandir(samples_root) as it:
        for entry in it:
            if entry.is_dir():
                out[entry.name] = Path(entry.path)
    return out


def load_resolved_tar_paths(resolved_json: str) -> List[str]:
    if not resolved_json:
        return []
    p = Path(resolved_json)
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    arr: List[str] = []
    if isinstance(data, dict):
        resolved_files = data.get("resolved_files", [])
        if isinstance(resolved_files, list):
            for x in resolved_files:
                if isinstance(x, str):
                    arr.append(x)
                elif isinstance(x, dict):
                    v = x.get("path") or x.get("file") or x.get("tar")
                    if isinstance(v, str):
                        arr.append(v)
    elif isinstance(data, list):
        for x in data:
            if isinstance(x, str):
                arr.append(x)

    dedup = []
    seen = set()
    for x in arr:
        xp = str(Path(x).resolve())
        if xp not in seen:
            seen.add(xp)
            dedup.append(xp)
    return dedup


def count_images_in_tar(tar_path: str) -> int:
    n = 0
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                ext = Path(member.name).suffix.lower()
                if ext in IMAGE_EXTS:
                    n += 1
    except Exception:
        return 0
    return n


def save_image_bytes_as_turn0(data: bytes, ext: str, out_path: Path) -> bool:
    try:
        ext = ext.lower()
        if ext in (".jpg", ".jpeg"):
            out_path.write_bytes(data)
            return True

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(out_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def process_one_tar(
    tar_path: str,
    samples_root: Path,
    existing_ids: Set[str],
    id_lock: threading.Lock,
) -> Dict[str, Any]:
    stats = {"tar_path": tar_path, "images": 0, "written": 0, "skipped": 0, "errors": 0}

    tar_file = Path(tar_path)
    if not tar_file.exists():
        stats["errors"] += 1
        return stats

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                ext = Path(name).suffix.lower()
                if ext not in IMAGE_EXTS:
                    continue

                stats["images"] += 1
                sid = Path(name).stem
                if not sid:
                    stats["skipped"] += 1
                    continue

                with id_lock:
                    if sid in existing_ids:
                        stats["skipped"] += 1
                        continue
                    existing_ids.add(sid)

                try:
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        raise ValueError("tar.extractfile is None")
                    data = file_obj.read()

                    sample_dir = samples_root / sid
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    turn0_path = sample_dir / "turn_0.jpg"

                    ok = save_image_bytes_as_turn0(data, ext, turn0_path)
                    if not ok:
                        with id_lock:
                            existing_ids.discard(sid)
                        stats["errors"] += 1
                        continue

                    sample_json = sample_dir / "sample.json"
                    meta = load_json(sample_json, default={})
                    meta.setdefault("sample_id", sid)
                    source = meta.setdefault("source", {})
                    source.setdefault("image_path", f"tar://{tar_path}::{member.name}")
                    source.setdefault("caption", "")
                    source.setdefault("tar_path", tar_path)
                    source.setdefault("tar_member", member.name)
                    meta.setdefault("rounds", [])
                    save_json(sample_json, meta)

                    stats["written"] += 1
                except Exception:
                    with id_lock:
                        existing_ids.discard(sid)
                    stats["errors"] += 1
    except Exception:
        stats["errors"] += 1

    return stats


class PromptClient:
    def __init__(
        self,
        api_url: str,
        api_token: str,
        trace_id: str,
        model_name: str,
        max_retries: int,
        request_timeout: int,
    ) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.headers = {
            "Accept-Encoding": "gzip, deflate, br",
            "Authorization": f"Bearer {api_token}",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Trace-id": trace_id,
        }

    @staticmethod
    def _encode_image(image_path: str) -> Tuple[Optional[str], Optional[str]]:
        p = Path(image_path)
        if not p.exists():
            return None, None
        mime_type, _ = mimetypes.guess_type(str(p))
        if not mime_type:
            mime_type = "image/png"
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type

    def generate_candidates(
        self,
        image_path: str,
        caption: str,
        turn: int,
        candidate_num: int,
        target_categories: List[str],
    ) -> List[Dict[str, Any]]:
        img_base64, mime_type = self._encode_image(image_path)
        if not img_base64:
            return []

        user_prompt = (
            f"{CONSTRUCTION_LOGIC}\n\n"
            f"### INPUT DATA\n"
            f"Image Caption: \"{caption}\"\n"
            f"Current Turn: {turn}\n"
            f"Candidate Num: {candidate_num}\n"
            f"Allowed Categories: {', '.join(target_categories)}\n"
            f"Analyze the current image and provide editing options in JSON."
        )

        payload = {
            "model": self.model_name,
            "contents": {
                "role": "USER",
                "parts": [
                    {"inline_data": {"mimeType": mime_type, "data": img_base64}},
                    {"text": user_prompt},
                ],
            },
            "stream": False,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.request_timeout)
                if resp.status_code == 200:
                    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return parse_response_text_to_json_list(content)

                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(20, (2 ** attempt) * 0.5 + random.uniform(0.2, 0.8)))
                    continue
                time.sleep(0.5)
            except Exception:
                time.sleep(min(20, (2 ** attempt) * 0.5 + random.uniform(0.2, 0.8)))
        return []


class NanoImageClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_retries: int,
        request_timeout: int,
        jpeg_quality: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def pick_best_resolution(w: int, h: int) -> Dict[str, Any]:
        src_ratio = w / h
        best = SUPPORTED_RESOLUTIONS[0]
        min_diff = float("inf")
        for res in SUPPORTED_RESOLUTIONS:
            diff = abs(src_ratio - (res["width"] / res["height"]))
            if diff < min_diff:
                min_diff = diff
                best = res
        return best

    @staticmethod
    def resize_and_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        w, h = img.size
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    @staticmethod
    def img_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt, quality=quality)
        return buf.getvalue()

    def generate_one(self, input_image: Path, prompt: str) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        with Image.open(input_image) as raw_img:
            res = self.pick_best_resolution(*raw_img.size)
            pre_img = self.resize_and_crop(raw_img, res["width"], res["height"])
            pre_bytes = self.img_to_bytes(pre_img, "JPEG", quality=self.jpeg_quality)

        in_b64 = base64.b64encode(pre_bytes).decode("utf-8")

        payload = {
            "model": self.model_name,
            "contents": {
                "role": "USER",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": in_b64}},
                    {"text": prompt},
                ],
            },
            "generation_config": {
                "response_modalities": ["IMAGE"],
                "image_config": {
                    "aspect_ratio": res["aspect_ratio"],
                    "image_output_options": {"mime_type": "image/jpeg"},
                },
            },
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/images/gemini_flash/generations"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
                if resp.status_code == 429:
                    time.sleep(min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.1, 0.9)))
                    continue

                resp.raise_for_status()
                data = resp.json()
                candidate = data.get("candidates", [{}])[0]
                part = candidate.get("content", {}).get("parts", [{}])[0]
                inline = part.get("inlineData") or part.get("inline_data") or {}

                if inline.get("data"):
                    out_bytes = base64.b64decode(inline["data"])
                    return out_bytes, res

                time.sleep(0.5)
            except Exception:
                time.sleep(min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.1, 0.9)))

        return None, None


class UnifiedScheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.samples_root = Path(args.samples_root).resolve()
        self.dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else self.samples_root.parent
        self.tasks_root = self.dataset_root / "tasks"
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        self.samples_root.mkdir(parents=True, exist_ok=True)

        self.cache_file = Path(args.cache_file).resolve() if args.cache_file else (self.tasks_root / "auto_scheduler_cache.json")
        self.cache = load_json(self.cache_file, default={})

        self.target_categories = [x.strip() for x in args.target_categories.split(",") if x.strip()]
        if not self.target_categories:
            self.target_categories = ALL_CATEGORIES.copy()

        self.prompt_client = PromptClient(
            api_url=args.prompt_api_url,
            api_token=args.prompt_api_token,
            trace_id=args.prompt_trace_id,
            model_name=args.prompt_model_name,
            max_retries=args.prompt_max_retries,
            request_timeout=args.prompt_request_timeout,
        )
        self.nano_client = NanoImageClient(
            base_url=args.image_base_url,
            api_key=args.image_api_key,
            model_name=args.image_model_name,
            max_retries=args.image_max_retries,
            request_timeout=args.image_request_timeout,
            jpeg_quality=args.nano_jpeg_quality,
        )

        self.state_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.category_lock = threading.Lock()
        self.id_lock = threading.Lock()

        self.states: Dict[str, Dict[str, Any]] = {}
        self.existing_sample_ids: Set[str] = set()

        self.prompt_done_by_turn: Dict[int, int] = {t: 0 for t in range(1, args.max_turn + 1)}
        self.image_done_by_turn: Dict[int, int] = {t: 0 for t in range(1, args.max_turn + 1)}

        self.category_counts = {c: 0 for c in self.target_categories}
        self.active_counts = {c: 0 for c in self.target_categories}

        self.tar_queue: "queue.Queue[str]" = queue.Queue()
        self.prompt_queue: "queue.PriorityQueue[Tuple[int, int, str]]" = queue.PriorityQueue()
        self.image_queue: "queue.PriorityQueue[Tuple[int, int, str]]" = queue.PriorityQueue()
        self.dispatch_seq = itertools.count()

        self.inflight_tars: Set[str] = set()
        self.inflight_prompt: Set[str] = set()
        self.inflight_image: Set[str] = set()

        self.tar_feed_done = False
        self.tar_meta: Dict[str, Dict[str, Any]] = {}
        self.tar_limit_reached = False
        self.tar_skip_remaining = max(0, int(args.tar_skip_images))
        self.tar_skipped_images = 0

        self.stop_event = threading.Event()
        self.last_cache_save_time = 0.0

    def save_cache(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_cache_save_time < self.args.cache_flush_interval):
            return
        data = {
            "dataset_root": str(self.dataset_root),
            "samples_root": str(self.samples_root),
            "max_turn": self.args.max_turn,
            "sample_states": self.states,
            "prompt_done_by_turn": self.prompt_done_by_turn,
            "image_done_by_turn": self.image_done_by_turn,
            "category_counts": self.category_counts,
            "bootstrap_tars": self.cache.get("bootstrap_tars", {}),
            "tar_feed_done": self.tar_feed_done,
            "tar_limit_reached": self.tar_limit_reached,
            "tar_skip_images": self.args.tar_skip_images,
            "tar_skipped_images": self.tar_skipped_images,
            "timestamp": now,
        }
        save_json(self.cache_file, data)
        self.last_cache_save_time = now

    def bootstrap_from_tars(self) -> None:
        tar_paths = load_resolved_tar_paths(self.args.resolved_json)
        bootstrap_cache = self.cache.get("bootstrap_tars", {})
        if not isinstance(bootstrap_cache, dict):
            bootstrap_cache = {}
            self.cache["bootstrap_tars"] = bootstrap_cache

        if not tar_paths:
            print("[bootstrap] 跳过：未提供有效 resolved_json 或 resolved_files 为空")
            self.tar_feed_done = True
            self.save_cache(force=True)
            return

        if self.args.max_tar_samples > 0 and len(self.existing_sample_ids) >= self.args.max_tar_samples:
            self.tar_limit_reached = True
            self.tar_feed_done = True
            print(
                f"[bootstrap] 跳过：现有样本数={len(self.existing_sample_ids)} 已达到上限 max_tar_samples={self.args.max_tar_samples}"
            )
            self.save_cache(force=True)
            return

        skipped = 0
        enqueued = 0
        missing = 0
        skipped_due_to_prefix = 0
        todo: List[str] = []

        for tar_path in tar_paths:
            p = Path(tar_path)
            if not p.exists():
                missing += 1
                continue

            st = p.stat()
            key = str(p.resolve())

            if self.tar_skip_remaining > 0:
                c = count_images_in_tar(key)
                self.tar_skipped_images += c
                self.tar_skip_remaining = max(0, self.tar_skip_remaining - c)
                skipped_due_to_prefix += 1
                continue

            old = bootstrap_cache.get(key, {})
            if (
                (not self.args.force_rebootstrap)
                and isinstance(old, dict)
                and float(old.get("mtime", -1)) == float(st.st_mtime)
                and int(old.get("size", -1)) == int(st.st_size)
                and bool(old.get("ok", False))
            ):
                skipped += 1
                continue

            self.tar_meta[key] = {"mtime": st.st_mtime, "size": st.st_size}
            todo.append(key)

        for tar_path in todo:
            self.tar_queue.put(tar_path)
            enqueued += 1

        self.tar_feed_done = True
        print(
            f"[bootstrap] total_tars={len(tar_paths)}, enqueue={enqueued}, "
            f"skip_unchanged={skipped}, skip_prefix_tars={skipped_due_to_prefix}, "
            f"skip_prefix_images={self.tar_skipped_images}, missing={missing}"
        )
        self.save_cache(force=True)

    def refresh_sample_state(
        self,
        sample_id: str,
        sample_dir: Path,
        prev_state: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if prev_state is None:
            prev_state = {}

        sample_json = sample_dir / "sample.json"
        mtime = sample_json.stat().st_mtime if sample_json.exists() else 0.0

        if (not force) and prev_state and float(prev_state.get("sample_json_mtime", -1)) == float(mtime):
            return prev_state

        meta = load_json(sample_json, default={})
        prompt_turns = sorted(parse_prompt_turns(meta))
        prompt_turn_set = set(prompt_turns)

        max_image_turn = find_contiguous_max_image_turn(sample_dir, max_turn=self.args.max_turn)
        next_turn = max_image_turn + 1

        failed = bool(prev_state.get("failed", False))
        stall_count = int(prev_state.get("stall_count", 0))

        if failed:
            status = "failed"
        elif max_image_turn >= self.args.max_turn:
            status = "done"
        elif next_turn in prompt_turn_set:
            status = "need_image"
        else:
            status = "need_prompt"

        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sample_json_mtime": mtime,
            "max_image_turn": max_image_turn,
            "next_turn": next_turn,
            "prompt_turns": prompt_turns,
            "status": status,
            "failed": failed,
            "stall_count": stall_count,
        }

    def _enqueue_one_locked(self, sid: str, st: Dict[str, Any]) -> None:
        status = st.get("status")
        if status in ("done", "failed"):
            return

        next_turn = int(st.get("next_turn", 0))
        prio = -next_turn
        seq = next(self.dispatch_seq)

        if status == "need_prompt":
            if sid not in self.inflight_prompt:
                self.inflight_prompt.add(sid)
                self.prompt_queue.put((prio, seq, sid))
        elif status == "need_image":
            if sid not in self.inflight_image:
                self.inflight_image.add(sid)
                self.image_queue.put((prio, seq, sid))

    def _register_new_sample_locked(self, sid: str, sample_dir: Path) -> None:
        prev = self.states.get(sid, {})
        st = self.refresh_sample_state(sample_id=sid, sample_dir=sample_dir, prev_state=prev, force=True)
        self.states[sid] = st
        self.existing_sample_ids.add(sid)
        self._enqueue_one_locked(sid, st)

    def init_states(self) -> None:
        old_states_raw = self.cache.get("sample_states", {})
        old_states = old_states_raw if isinstance(old_states_raw, dict) else {}

        sample_dirs = discover_samples(self.samples_root)
        states: Dict[str, Dict[str, Any]] = {}
        prompt_done = {t: 0 for t in range(1, self.args.max_turn + 1)}
        image_done = {t: 0 for t in range(1, self.args.max_turn + 1)}

        for sid, sample_dir in sample_dirs.items():
            prev = old_states.get(sid, {})
            if not isinstance(prev, dict):
                prev = {}
            st = self.refresh_sample_state(sid, sample_dir, prev_state=prev, force=True)
            states[sid] = st

            for t in st.get("prompt_turns", []):
                if 1 <= int(t) <= self.args.max_turn:
                    prompt_done[int(t)] += 1

            max_img_turn = int(st.get("max_image_turn", -1))
            for t in range(1, min(max_img_turn, self.args.max_turn) + 1):
                image_done[t] += 1

        self.states = states
        self.existing_sample_ids = set(states.keys())
        self.prompt_done_by_turn = prompt_done
        self.image_done_by_turn = image_done

        for sid, sample_dir in sample_dirs.items():
            meta = load_json(sample_dir / "sample.json", default={})
            rounds = meta.get("rounds", [])
            if not isinstance(rounds, list):
                continue
            for r in rounds:
                if not isinstance(r, dict):
                    continue
                p = r.get("prompt", {})
                if not isinstance(p, dict):
                    continue
                cat = p.get("category")
                if cat in self.category_counts:
                    self.category_counts[cat] += 1

    def _compute_progress_snapshot_locked(self) -> Dict[str, Any]:
        total = len(self.states)
        done = 0
        failed = 0
        need_prompt = 0
        need_image = 0
        for st in self.states.values():
            s = st.get("status")
            if s == "done":
                done += 1
            elif s == "failed":
                failed += 1
            elif s == "need_prompt":
                need_prompt += 1
            elif s == "need_image":
                need_image += 1

        return {
            "total": total,
            "done": done,
            "failed": failed,
            "need_prompt": need_prompt,
            "need_image": need_image,
            "qt": self.tar_queue.qsize(),
            "qp": self.prompt_queue.qsize(),
            "qi": self.image_queue.qsize(),
            "it": len(self.inflight_tars),
            "ip": len(self.inflight_prompt),
            "ii": len(self.inflight_image),
            "prompt_done_by_turn": dict(self.prompt_done_by_turn),
            "image_done_by_turn": dict(self.image_done_by_turn),
        }

    def _format_turn_counts(self, d: Dict[int, int]) -> str:
        return " ".join([f"{t}:{d.get(t, 0)}" for t in range(1, self.args.max_turn + 1)])

    def _progress_printer_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.state_lock:
                s = self._compute_progress_snapshot_locked()

            line = (
                f"\r[D:{s['done']}/{s['total']} F:{s['failed']} "
                f"NP:{s['need_prompt']} NI:{s['need_image']} "
                f"QT:{s['qt']} QP:{s['qp']} QI:{s['qi']} "
                f"IT:{s['it']} IP:{s['ip']} II:{s['ii']}] "
                f"P[{self._format_turn_counts(s['prompt_done_by_turn'])}] "
                f"I[{self._format_turn_counts(s['image_done_by_turn'])}]"
            )
            print(line, end="", flush=True)
            time.sleep(self.args.progress_interval)
        print()

    def _calc_progressed(self, action: str, before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, int]:
        expected_turn = int(before.get("next_turn", -1))
        if expected_turn < 1:
            return False, expected_turn

        if action == "prompt":
            old_set = set(before.get("prompt_turns", []))
            new_set = set(after.get("prompt_turns", []))
            return ((expected_turn not in old_set) and (expected_turn in new_set)), expected_turn

        old_max = int(before.get("max_image_turn", -1))
        new_max = int(after.get("max_image_turn", -1))
        return ((old_max < expected_turn) and (new_max >= expected_turn)), expected_turn

    def _update_state_after_action(self, sid: str, action: str, before: Dict[str, Any], result: str) -> None:
        sample_dir = Path(before["sample_dir"])
        new_state = self.refresh_sample_state(sid, sample_dir, prev_state=before, force=True)

        progressed, turn = self._calc_progressed(action, before, new_state)

        if progressed:
            new_state["stall_count"] = 0
            if 1 <= turn <= self.args.max_turn:
                if action == "prompt":
                    self.prompt_done_by_turn[turn] += 1
                else:
                    self.image_done_by_turn[turn] += 1
        else:
            if before.get("status") in ("need_prompt", "need_image"):
                if result.startswith("fail") or result.startswith("skip_"):
                    sc = int(before.get("stall_count", 0)) + 1
                    new_state["stall_count"] = sc
                    if sc >= self.args.fail_after_attempts:
                        new_state["failed"] = True
                        new_state["status"] = "failed"
                else:
                    new_state["stall_count"] = int(before.get("stall_count", 0))

        self.states[sid] = new_state
        self._enqueue_one_locked(sid, new_state)

    def _generate_single_prompt_for_sample(self, sid: str, before: Dict[str, Any]) -> str:
        sample_dir = Path(before["sample_dir"])
        sample_json = sample_dir / "sample.json"
        turn = int(before["next_turn"])
        prev_img = sample_dir / f"turn_{turn - 1}.jpg"

        if not prev_img.exists():
            return "skip_missing_prev_image"

        with self.file_lock:
            meta = load_json(sample_json, default={})
            round_meta = get_round(meta, turn, create=True)
            existing_prompt = ""
            if isinstance(round_meta, dict):
                p = round_meta.get("prompt", {})
                if isinstance(p, dict):
                    existing_prompt = str(p.get("edit_prompt", "")).strip()
            if existing_prompt:
                return "exists"

        source = meta.get("source", {}) if isinstance(meta, dict) else {}
        caption = str(source.get("caption", ""))

        candidates = self.prompt_client.generate_candidates(
            image_path=str(prev_img),
            caption=caption,
            turn=turn,
            candidate_num=self.args.prompt_candidate_num,
            target_categories=self.target_categories,
        )
        if not candidates:
            return "fail_no_candidates"

        with self.category_lock:
            valid_candidates = [
                c for c in candidates
                if c.get("category") in self.category_counts and str(c.get("edit_prompt", "")).strip()
            ]
            if not valid_candidates:
                return "fail_no_valid_category"

            valid_candidates.sort(key=lambda x: self.category_counts[x["category"]] + self.active_counts[x["category"]])
            best = valid_candidates[0]
            selected_cat = best["category"]
            self.active_counts[selected_cat] += 1

        try:
            prompt_text = str(best.get("edit_prompt", "")).strip()
            if not prompt_text:
                return "fail_empty_prompt"

            with self.file_lock:
                meta = load_json(sample_json, default={})
                round_meta = get_round(meta, turn, create=True)
                round_meta["prompt"] = {
                    "category": selected_cat,
                    "edit_prompt": prompt_text,
                    "reasoning": best.get("reasoning", ""),
                    "candidate_num": self.args.prompt_candidate_num,
                    "target_categories": self.target_categories,
                    "timestamp": time.time(),
                }
                round_meta["status"] = "prompt_ready"
                save_json(sample_json, meta)

            with self.category_lock:
                self.category_counts[selected_cat] += 1
            return "success"
        finally:
            with self.category_lock:
                self.active_counts[selected_cat] -= 1

    def _generate_single_image_for_sample(self, sid: str, before: Dict[str, Any]) -> str:
        sample_dir = Path(before["sample_dir"])
        sample_json = sample_dir / "sample.json"
        turn = int(before["next_turn"])

        input_image = sample_dir / f"turn_{turn - 1}.jpg"
        output_image = sample_dir / f"turn_{turn}.jpg"

        if not input_image.exists():
            return "skip_missing_input"
        if output_image.exists():
            return "exists"

        with self.file_lock:
            meta = load_json(sample_json, default={})
            round_meta = get_round(meta, turn, create=False)
            round_prompt = ""
            if isinstance(round_meta, dict):
                p = round_meta.get("prompt", {})
                if isinstance(p, dict):
                    round_prompt = str(p.get("edit_prompt", "")).strip()
            if not round_prompt:
                return "skip_missing_instruction"

        out_bytes, res_meta = self.nano_client.generate_one(input_image=input_image, prompt=round_prompt)
        if not out_bytes:
            return "fail_api"

        output_image.write_bytes(out_bytes)

        with self.file_lock:
            meta = load_json(sample_json, default={})
            round_meta = get_round(meta, turn, create=True)
            round_meta.setdefault("prompt", {"edit_prompt": round_prompt})
            round_meta["image"] = {
                "generator": "nano",
                "model_name": self.nano_client.model_name,
                "base_url": self.nano_client.base_url,
                "input_image": str(input_image),
                "output_image": str(output_image),
                "resolution": res_meta,
                "params": {
                    "max_retries": self.nano_client.max_retries,
                    "request_timeout": self.nano_client.request_timeout,
                    "jpeg_quality": self.nano_client.jpeg_quality,
                },
                "timestamp": time.time(),
            }
            round_meta["status"] = "image_ready"
            save_json(sample_json, meta)

        return "success"

    def _process_tar_stream(self, tar_path: str) -> Dict[str, Any]:
        stats = {"images": 0, "written": 0, "skipped": 0, "errors": 0}
        tar_file = Path(tar_path)
        if not tar_file.exists():
            stats["errors"] += 1
            return stats

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar:
                    if self.stop_event.is_set() or self.tar_limit_reached:
                        break
                    if not member.isfile():
                        continue

                    name = Path(member.name).name
                    ext = Path(name).suffix.lower()
                    if ext not in IMAGE_EXTS:
                        continue

                    limit_hit = False
                    with self.id_lock:
                        if self.args.max_tar_samples > 0 and len(self.existing_sample_ids) >= self.args.max_tar_samples:
                            self.tar_limit_reached = True
                            limit_hit = True
                    if limit_hit:
                        break

                    while (
                        self.prompt_queue.qsize() >= self.args.prompt_queue_high_watermark
                        and not self.stop_event.is_set()
                        and not self.tar_limit_reached
                    ):
                        time.sleep(self.args.backpressure_sleep)

                    if self.stop_event.is_set() or self.tar_limit_reached:
                        break

                    stats["images"] += 1
                    sid = Path(name).stem
                    if not sid:
                        stats["skipped"] += 1
                        continue

                    with self.id_lock:
                        if self.args.max_tar_samples > 0 and len(self.existing_sample_ids) >= self.args.max_tar_samples:
                            self.tar_limit_reached = True
                            limit_hit = True
                        elif sid in self.existing_sample_ids:
                            stats["skipped"] += 1
                            continue
                        else:
                            self.existing_sample_ids.add(sid)
                    if limit_hit:
                        break

                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            raise ValueError("tar.extractfile returned None")
                        data = file_obj.read()

                        sample_dir = self.samples_root / sid
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        turn0_path = sample_dir / "turn_0.jpg"

                        ok = save_image_bytes_as_turn0(data, ext, turn0_path)
                        if not ok:
                            with self.id_lock:
                                self.existing_sample_ids.discard(sid)
                            stats["errors"] += 1
                            continue

                        with self.file_lock:
                            sample_json = sample_dir / "sample.json"
                            meta = load_json(sample_json, default={})
                            meta.setdefault("sample_id", sid)
                            source = meta.setdefault("source", {})
                            source.setdefault("image_path", f"tar://{tar_path}::{member.name}")
                            source.setdefault("caption", "")
                            source.setdefault("tar_path", tar_path)
                            source.setdefault("tar_member", member.name)
                            meta.setdefault("rounds", [])
                            save_json(sample_json, meta)

                        with self.state_lock:
                            self._register_new_sample_locked(sid, sample_dir)

                        stats["written"] += 1
                    except Exception:
                        with self.id_lock:
                            self.existing_sample_ids.discard(sid)
                        stats["errors"] += 1
        except Exception:
            stats["errors"] += 1

        return stats

    def _tar_worker_loop(self) -> None:
        while True:
            if self.stop_event.is_set() and self.tar_queue.empty():
                return

            try:
                tar_path = self.tar_queue.get(timeout=0.5)
            except queue.Empty:
                if self.tar_feed_done and self.tar_queue.empty():
                    return
                continue

            skip_due_to_limit = False
            with self.state_lock:
                if self.tar_limit_reached:
                    skip_due_to_limit = True
                else:
                    self.inflight_tars.add(tar_path)

            if skip_due_to_limit:
                self.tar_queue.task_done()
                if self.tar_feed_done and self.tar_queue.empty():
                    return
                continue

            result = self._process_tar_stream(tar_path)

            with self.state_lock:
                self.inflight_tars.discard(tar_path)
                bootstrap_cache = self.cache.get("bootstrap_tars", {})
                if not isinstance(bootstrap_cache, dict):
                    bootstrap_cache = {}
                    self.cache["bootstrap_tars"] = bootstrap_cache

                meta = self.tar_meta.get(tar_path, {})
                bootstrap_cache[tar_path] = {
                    "mtime": meta.get("mtime", 0),
                    "size": meta.get("size", 0),
                    "ok": result.get("errors", 0) == 0,
                    "images": result.get("images", 0),
                    "written": result.get("written", 0),
                    "skipped": result.get("skipped", 0),
                    "errors": result.get("errors", 0),
                    "timestamp": time.time(),
                }

            self.tar_queue.task_done()

    def _prompt_worker_loop(self) -> None:
        while True:
            if self.stop_event.is_set() and self.prompt_queue.empty():
                return
            try:
                _, _, sid = self.prompt_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self.state_lock:
                before = dict(self.states.get(sid, {}))

            result = "skip_no_state"
            if before and before.get("status") == "need_prompt":
                try:
                    result = self._generate_single_prompt_for_sample(sid, before)
                except Exception:
                    result = "fail_exception"

            with self.state_lock:
                self.inflight_prompt.discard(sid)
                if before:
                    self._update_state_after_action(sid, "prompt", before, result)

            self.prompt_queue.task_done()

    def _image_worker_loop(self) -> None:
        while True:
            if self.stop_event.is_set() and self.image_queue.empty():
                return
            try:
                _, _, sid = self.image_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self.state_lock:
                before = dict(self.states.get(sid, {}))

            result = "skip_no_state"
            if before and before.get("status") == "need_image":
                try:
                    result = self._generate_single_image_for_sample(sid, before)
                except Exception:
                    result = "fail_exception"

            with self.state_lock:
                self.inflight_image.discard(sid)
                if before:
                    self._update_state_after_action(sid, "image", before, result)

            self.image_queue.task_done()

    def run(self) -> None:
        self.init_states()

        tar_threads = [
            threading.Thread(target=self._tar_worker_loop, daemon=True)
            for _ in range(max(1, self.args.tar_workers))
        ]
        prompt_threads = [
            threading.Thread(target=self._prompt_worker_loop, daemon=True)
            for _ in range(max(1, self.args.prompt_workers))
        ]
        image_threads = [
            threading.Thread(target=self._image_worker_loop, daemon=True)
            for _ in range(max(1, self.args.image_workers))
        ]
        progress_thread = threading.Thread(target=self._progress_printer_loop, daemon=True)

        for t in tar_threads + prompt_threads + image_threads:
            t.start()
        progress_thread.start()

        self.bootstrap_from_tars()

        try:
            for loop_idx in range(self.args.max_loops):
                with self.state_lock:
                    if self.args.dispatch_refresh:
                        for sid, st in list(self.states.items()):
                            self.states[sid] = self.refresh_sample_state(
                                sample_id=sid,
                                sample_dir=Path(st["sample_dir"]),
                                prev_state=st,
                                force=False,
                            )

                    for sid, st in list(self.states.items()):
                        self._enqueue_one_locked(sid, st)

                    snap = self._compute_progress_snapshot_locked()
                    tar_done = self.tar_feed_done and self.tar_queue.empty() and len(self.inflight_tars) == 0
                    all_terminal = snap["total"] > 0 and (snap["done"] + snap["failed"] == snap["total"])
                    no_work = (
                        self.prompt_queue.empty()
                        and self.image_queue.empty()
                        and len(self.inflight_prompt) == 0
                        and len(self.inflight_image) == 0
                    )
                    no_samples_and_tar_done = (snap["total"] == 0 and tar_done)

                if (tar_done and all_terminal and no_work) or (no_samples_and_tar_done and no_work):
                    self.stop_event.set()
                    break

                if loop_idx % self.args.cache_flush_every_loops == 0:
                    with self.state_lock:
                        self.save_cache(force=False)

                time.sleep(self.args.dispatch_interval)
            else:
                self.stop_event.set()

            self.tar_queue.join()
            self.prompt_queue.join()
            self.image_queue.join()

        finally:
            self.stop_event.set()
            for t in tar_threads + prompt_threads + image_threads:
                t.join(timeout=2.0)
            progress_thread.join(timeout=2.0)

            with self.state_lock:
                self.save_cache(force=True)
                final_snap = self._compute_progress_snapshot_locked()

            print(
                f"[final] done={final_snap['done']}/{final_snap['total']}, "
                f"failed={final_snap['failed']}, "
                f"prompt_by_turn={self.prompt_done_by_turn}, "
                f"image_by_turn={self.image_done_by_turn}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified auto scheduler: streaming tar + prompt/image dual pools.")
    parser.add_argument("--samples_root", type=str, required=True, help="sample保存目录")
    parser.add_argument("--dataset_root", type=str, default="", help="可选；不传则自动取 samples_root 父目录")
    parser.add_argument("--resolved_json", type=str, default="", help="包含resolved_files的json文件路径")
    parser.add_argument("--max_turn", type=int, default=5)
    parser.add_argument("--cache_file", type=str, default="")
    parser.add_argument("--cache_flush_interval", type=float, default=5.0)
    parser.add_argument("--cache_flush_every_loops", type=int, default=20)

    parser.add_argument("--max_loops", type=int, default=1000000)
    parser.add_argument("--dispatch_interval", type=float, default=0.1)
    parser.add_argument("--dispatch_refresh", action="store_true")
    parser.add_argument("--progress_interval", type=float, default=1.0)
    parser.add_argument("--fail_after_attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--tar_workers", type=int, default=8)
    parser.add_argument("--force_rebootstrap", action="store_true")
    parser.add_argument("--max_tar_samples", type=int, default=0, help="tar侧读取样本总上限；<=0表示不限制")
    parser.add_argument("--tar_skip_images", type=int, default=0, help="按tar顺序跳过前K张图像（按整tar快速累计）")
    parser.add_argument("--prompt_queue_high_watermark", type=int, default=2000)
    parser.add_argument("--backpressure_sleep", type=float, default=0.2)

    parser.add_argument("--prompt_api_url", type=str, default="YOUR_API_URL")
    parser.add_argument("--prompt_api_token", type=str, default="YOUR_API_TOKEN")
    parser.add_argument("--prompt_trace_id", type=str, default="gen-edit-task")
    parser.add_argument("--prompt_model_name", type=str, default="YOUR_MODEL_NAME")
    parser.add_argument("--prompt_workers", type=int, default=64)
    parser.add_argument("--prompt_max_retries", type=int, default=8)
    parser.add_argument("--prompt_request_timeout", type=int, default=90)
    parser.add_argument("--prompt_candidate_num", type=int, default=3)
    parser.add_argument("--target_categories", type=str, default=",".join(ALL_CATEGORIES))

    parser.add_argument("--image_base_url", type=str, default="YOUR_API_URL")
    parser.add_argument("--image_api_key", type=str, default="YOUR_API_KEY")
    parser.add_argument("--image_model_name", type=str, default="YOUR_IMAGE_MODEL_NAME")
    parser.add_argument("--image_workers", type=int, default=16)
    parser.add_argument("--image_max_retries", type=int, default=8)
    parser.add_argument("--image_request_timeout", type=int, default=300)
    parser.add_argument("--nano_jpeg_quality", type=int, default=95)

    args = parser.parse_args()

    if args.max_turn < 1:
        raise ValueError("--max_turn 必须 >= 1")

    random.seed(args.seed)

    scheduler = UnifiedScheduler(args)
    scheduler.run()


if __name__ == "__main__":
    main()