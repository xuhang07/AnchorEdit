# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# ================= 配置区域 =================
API_URL = "YOUR_API_URL"
API_TOKEN = "YOUR_API_TOKEN"
TRACE_ID = "gen-edit-task"
MODEL_NAME = "YOUR_MODEL_NAME"

INPUT_FILE = "/path/to/dataset.jsonl"

NUM_WORKERS = 1000
MAX_RETRIES = 1000

ALL_CATEGORIES = [
    "add", "remove", "replace", "extract", "adjust",
    "compose", "motion", "text", "background", "color",
    "beauty", "lowlevel", "stylize", "viewpoint", "portrait"
]

# ================= 构造逻辑知识库 (自主分析版 System Prompt) =================
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


def load_items(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            return []
        except json.JSONDecodeError:
            f.seek(0)
            items: List[Dict[str, Any]] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    continue
            return items


def normalize_item(item: Dict[str, Any]) -> Tuple[Optional[str], str]:
    image_path = item.get("image_path") or item.get("source_image") or item.get("reference_image")
    caption = item.get("caption_en") or item.get("caption") or item.get("original_caption") or ""
    if not image_path:
        return None, caption
    return str(image_path), str(caption)


def make_sample_id(item: Dict[str, Any], image_path: str) -> str:
    if item.get("sample_id"):
        return str(item["sample_id"])
    if item.get("id"):
        return str(item["id"])
    stem = Path(image_path).stem if image_path else "sample"
    raw = f"{image_path}|{item.get('caption_en','')}|{item.get('caption','')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
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


def write_json(path: Path, data: Dict[str, Any]) -> None:
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


def copy_or_download_image(src: str, dst: Path, timeout: int = 60) -> bool:
    try:
        if src.startswith("http://") or src.startswith("https://"):
            resp = requests.get(src, timeout=timeout)
            resp.raise_for_status()
            dst.write_bytes(resp.content)
            return True
        src_path = Path(src)
        if not src_path.exists():
            return False
        shutil.copy2(src_path, dst)
        return True
    except Exception:
        return False


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


class GeminiGenerator:
    def __init__(
        self,
        api_url: str,
        api_token: str,
        trace_id: str,
        model_name: str,
        max_retries: int,
        request_timeout: int,
        temperature: float,
    ) -> None:
        self.api_url = api_url
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.headers = {
            "Accept-Encoding": "gzip, deflate, br",
            "Authorization": f"Bearer {api_token}",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Trace-id": trace_id
        }
        self.model_name = model_name

    @staticmethod
    def _encode_image(image_path: str) -> Tuple[Optional[str], Optional[str]]:
        if not os.path.exists(image_path):
            return None, None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type

    def generate(
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
                    {"text": user_prompt}
                ]
            },
            "stream": False,
            "generation_config": {
                "temperature": self.temperature
            }
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.request_timeout)
                if response.status_code == 200:
                    content = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return parse_response_text_to_json_list(content)

                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(20, (2 ** attempt) * 0.5 + random.uniform(0.2, 0.8)))
                    continue
                time.sleep(0.5)
            except Exception:
                time.sleep(min(20, (2 ** attempt) * 0.5 + random.uniform(0.2, 0.8)))
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts for ONE specific editing turn.")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE, help="输入初始样本json/jsonl（需包含image_path或等价字段）")
    parser.add_argument("--dataset_root", type=str, default=str(Path(INPUT_FILE).parent), help="输出数据集根目录")
    parser.add_argument("--turn", type=int, required=True, help="当前轮次（从1开始）")
    parser.add_argument("--max_workers", type=int, default=128, help="并发线程数")
    parser.add_argument("--max_retries", type=int, default=8, help="API重试次数")
    parser.add_argument("--request_timeout", type=int, default=90, help="API超时秒数")
    parser.add_argument("--candidate_num", type=int, default=3, help="每张图生成候选prompt数量")
    parser.add_argument("--temperature", type=float, default=0.6, help="LLM采样温度")
    parser.add_argument("--seed", type=int, default=1234, help="shuffle随机种子")
    parser.add_argument("--target_categories", type=str, default=",".join(ALL_CATEGORIES), help="允许类别，逗号分隔")
    parser.add_argument("--api_url", type=str, default=os.getenv("PROMPT_API_URL", API_URL))
    parser.add_argument("--api_token", type=str, default=os.getenv("PROMPT_API_TOKEN", API_TOKEN))
    parser.add_argument("--trace_id", type=str, default=os.getenv("PROMPT_TRACE_ID", TRACE_ID))
    parser.add_argument("--model_name", type=str, default=os.getenv("PROMPT_MODEL_NAME", MODEL_NAME))
    args = parser.parse_args()

    if args.turn < 1:
        raise ValueError("--turn 必须 >= 1")
    if not args.api_token:
        raise ValueError("缺少API Token，请通过 --api_token 或环境变量 PROMPT_API_TOKEN 提供")

    random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    samples_root = dataset_root / "samples"
    tasks_root = dataset_root / "tasks"
    samples_root.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)

    task_file = tasks_root / f"turn_{args.turn:02d}_prompts.jsonl"
    target_categories = [x.strip() for x in args.target_categories.split(",") if x.strip()]
    if not target_categories:
        target_categories = ALL_CATEGORIES.copy()

    existing_task_ids = set()
    category_counts = {cat: 0 for cat in target_categories}
    active_counts = {cat: 0 for cat in target_categories}

    if task_file.exists():
        with open(task_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = str(obj.get("sample_id", ""))
                    cat = obj.get("category")
                    if sid:
                        existing_task_ids.add(sid)
                    if cat in category_counts:
                        category_counts[cat] += 1
                except Exception:
                    continue

    items = load_items(args.input_file)
    random.shuffle(items)

    build_stats = {
        "skip_missing_source": 0,
        "skip_missing_prev_turn": 0,
        "skip_current_turn_exists": 0,
        "skip_prompt_exists": 0,
        "skip_duplicate_sample": 0,
    }

    work_items: List[Dict[str, Any]] = []
    seen_sample_ids = set()

    for item in items:
        source_image, caption = normalize_item(item)
        if not source_image:
            build_stats["skip_missing_source"] += 1
            continue

        sample_id = make_sample_id(item, source_image)
        if sample_id in seen_sample_ids:
            build_stats["skip_duplicate_sample"] += 1
            continue
        seen_sample_ids.add(sample_id)

        sample_dir = samples_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_json = sample_dir / "sample.json"

        meta = read_json(sample_json, default={})
        meta.setdefault("sample_id", sample_id)
        meta.setdefault("source", {
            "image_path": source_image,
            "caption": caption,
            "raw_item": item,
        })
        meta.setdefault("rounds", [])

        turn0_path = sample_dir / "turn_0.jpg"
        if not turn0_path.exists():
            copied = copy_or_download_image(source_image, turn0_path)
            if not copied:
                build_stats["skip_missing_source"] += 1
                continue

        prev_img = sample_dir / f"turn_{args.turn - 1}.jpg"
        curr_img = sample_dir / f"turn_{args.turn}.jpg"

        if not prev_img.exists():
            build_stats["skip_missing_prev_turn"] += 1
            write_json(sample_json, meta)
            continue

        round_meta = get_round(meta, args.turn, create=False)

        if curr_img.exists():
            build_stats["skip_current_turn_exists"] += 1
            write_json(sample_json, meta)
            continue

        if sample_id in existing_task_ids:
            build_stats["skip_prompt_exists"] += 1
            write_json(sample_json, meta)
            continue

        if round_meta and round_meta.get("prompt", {}).get("edit_prompt"):
            build_stats["skip_prompt_exists"] += 1
            write_json(sample_json, meta)
            continue

        write_json(sample_json, meta)
        work_items.append({
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sample_json": str(sample_json),
            "input_image": str(prev_img),
            "output_image": str(curr_img),
            "caption": caption,
        })

    if not work_items:
        print("没有可处理样本：可能已完成当前轮，或缺少上一轮图像。")
        print("Build Stats:", json.dumps(build_stats, ensure_ascii=False))
        return

    generator = GeminiGenerator(
        api_url=args.api_url,
        api_token=args.api_token,
        trace_id=args.trace_id,
        model_name=args.model_name,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        temperature=args.temperature,
    )

    file_lock = threading.Lock()
    count_lock = threading.Lock()
    success_cnt = 0
    fail_cnt = 0
    written_ids = set(existing_task_ids)

    def process_one(work_item: Dict[str, Any]) -> bool:
        nonlocal success_cnt, fail_cnt

        sample_id = work_item["sample_id"]
        input_image = work_item["input_image"]
        caption = work_item["caption"]

        candidates = generator.generate(
            image_path=input_image,
            caption=caption,
            turn=args.turn,
            candidate_num=args.candidate_num,
            target_categories=target_categories,
        )
        if not candidates:
            with count_lock:
                fail_cnt += 1
            return False

        with count_lock:
            valid_candidates = [
                c for c in candidates
                if c.get("category") in category_counts and str(c.get("edit_prompt", "")).strip()
            ]
            if not valid_candidates:
                fail_cnt += 1
                return False
            valid_candidates.sort(
                key=lambda x: category_counts[x["category"]] + active_counts[x["category"]]
            )
            best = valid_candidates[0]
            selected_cat = best["category"]
            active_counts[selected_cat] += 1

        try:
            prompt_text = str(best.get("edit_prompt", "")).strip()
            if not prompt_text:
                with count_lock:
                    fail_cnt += 1
                return False

            sample_json = Path(work_item["sample_json"])
            with file_lock:
                meta = read_json(sample_json, default={})
                round_meta = get_round(meta, args.turn, create=True)
                round_meta["prompt"] = {
                    "category": selected_cat,
                    "edit_prompt": prompt_text,
                    "reasoning": best.get("reasoning", ""),
                    "candidate_num": args.candidate_num,
                    "target_categories": target_categories,
                    "generator_model": args.model_name,
                    "timestamp": time.time(),
                }
                round_meta["status"] = "prompt_ready"
                write_json(sample_json, meta)

                if sample_id not in written_ids:
                    task_obj = {
                        "sample_id": sample_id,
                        "turn": args.turn,
                        "sample_dir": work_item["sample_dir"],
                        "input_image": work_item["input_image"],
                        "output_image": work_item["output_image"],
                        "category": selected_cat,
                        "edit_prompt": prompt_text,
                        "reasoning": best.get("reasoning", ""),
                        "caption": caption,
                    }
                    with open(task_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(task_obj, ensure_ascii=False) + "\n")
                    written_ids.add(sample_id)

            with count_lock:
                category_counts[selected_cat] += 1
                success_cnt += 1
            return True

        finally:
            with count_lock:
                active_counts[selected_cat] -= 1

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_one, w) for w in work_items]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Turn {args.turn} prompt"):
            pass

    print(f"Turn {args.turn} prompt done. success={success_cnt}, fail={fail_cnt}, total={len(work_items)}")
    print("Build Stats:", json.dumps(build_stats, ensure_ascii=False))
    print(f"Task file: {task_file}")


if __name__ == "__main__":
    main()