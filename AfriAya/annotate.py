"""
annotate.py
Data Annotation Pipeline (Image Caption and Q&A):
1. Ensures images in a folder are sequentially named (image_01.jpg, image_02.jpg, …).
   - Only renames once; skips if files are already in correct format.
2. For each image, queries Gemini multimodal API to:
   - Generate an English caption
   - Translate caption to target language
   - Generate 3 Q&A pairs (descriptive, true/false, object/action)
   - Generate 1 multiple-choice question (3 options)
   - Return both English and target language versions
3. Writes results incrementally to JSON after each processed image.
4. Resumes from existing JSON data to avoid duplication.
Supports: Rate limit (for free tier), so far Gemini 2.0 flash seems to be the most stable it can allow upto 15 requests/minute → enforced with safe_sleep, you can adjust safe_sleep.
Usage:
  1. Ensure you have your images in one folder
  1. Set api to env with this command: export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
  2. Run script (example):
     python3 annotate.py \
        --images_dir ./krio_images \
        --out_json krio_img_captions_qa.json \
        --num_images 50 \
        --culture Krio \
        --target_language Krio \
        --original_query "Everyday Life in Sierra Leone"
Requires:
  pip install google-generativeai pillow
"""

import os, time, json, argparse, re, random
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# ---------- Gemini setup ----------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_ID)

# ---------- prompt builder ----------
def build_prompt(target_language: str) -> str:
    return f"""You are a helpful vision-language assistant.
1. Look at the image and write a clear, concise English caption (1-2 sentences).
2. Translate that caption into {target_language}.
3. Generate culturally relevant Q&A about the image:
   - Three questions with answers (English + {target_language}): one descriptive, one true/false, one object/action identification.
   - One multiple-choice question with 3 options (English + {target_language}).
Output in strict JSON with this schema, no explanations, no markdown, no code fences:
{{
  "caption": "English caption",
  "translated_caption": "{target_language} caption",
  "qa_pairs": [
    {{
      "type": "Descriptive",
      "question": "English question",
      "answer": "English answer",
      "translated_question": "{target_language} question",
      "translated_answer": "{target_language} answer"
    }},
    {{
      "type": "Contextual True/False",
      "question": "English question",
      "answer": "True/False",
      "translated_question": "{target_language} question",
      "translated_answer": "True/False in {target_language}"
    }},
    {{
      "type": "Object/Action Identification",
      "question": "English question",
      "answer": "English answer",
      "translated_question": "{target_language} question",
      "translated_answer": "{target_language} answer"
    }},
    {{
      "type": "Contextual Multiple Choice",
      "question": "English question",
      "answer": "Correct option in English",
      "translated_question": "{target_language} question",
      "translated_answer": "Correct option in {target_language}",
      "options": ["Option A (English)", "Option B (English)", "Option C (English)"],
      "translated_options": ["Option A ({target_language})", "Option B ({target_language})", "Option C ({target_language})"]
    }}
  ]
}}
"""

# ---------- helpers ----------
def ensure_sequential_names(images_dir: Path):
    valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    renamed = []
    for i, img in enumerate(images, start=1):
        expected_name = f"image_{i:02d}{img.suffix.lower()}"
        expected_path = images_dir / expected_name
        if img.name != expected_name:
            if not expected_path.exists():
                img.rename(expected_path)
        renamed.append(expected_path)
    return renamed

def load_image_resized(image_path: Path, max_size=1024):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def clean_json_text(raw_text: str) -> str:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```[a-zA-Z]*\s*", "", raw_text)
    raw_text = re.sub(r"```$", "", raw_text)
    raw_text = re.sub(r",\s*([}\]])", r"\1", raw_text)
    return raw_text.strip()

def safe_sleep(base=45, jitter=5):
    delay = base + random.uniform(-jitter, jitter)
    delay = max(40, delay)
    print(f"[INFO] Sleeping for {delay:.1f} seconds to avoid hitting rate limit…")
    time.sleep(delay)

def send_to_gemini(image_path: Path, prompt: str):
    img = load_image_resized(image_path)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        response = model.generate_content(
            [prompt, img],
            generation_config={"temperature": 0.3, "max_output_tokens": 8192},
            safety_settings=safety_settings
        )
    except Exception as e:
        print(f"[ERROR] Gemini request error for {image_path.name}: {e}")
        return None, None
    if not getattr(response, "candidates", None):
        print(f"[WARN] No candidates for {image_path.name}.")
        return None, None
    cand = response.candidates[0]
    content = getattr(cand, "content", None)
    if not content or not getattr(content, "parts", None):
        print(f"[WARN] No content.parts for {image_path.name}. Ratings: {cand.safety_ratings}")
        return None, None
    raw_text = None
    for part in content.parts:
        if getattr(part, "text", None):
            raw_text = part.text
            break
    if raw_text is None:
        print(f"[WARN] No textual part for {image_path.name}. Ratings: {cand.safety_ratings}")
        return None, None
    cleaned = clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error for {image_path.name}: {e}. Raw: {cleaned[:200]!r}")
        return None, raw_text
    return parsed, raw_text

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder with images to process")
    ap.add_argument("--out_json", default="img_captions_qa.json")
    ap.add_argument("--num_images", type=int, default=None, help="Max number of images to process")
    ap.add_argument("--culture", required=True, help="Culture name (e.g., Krio, Swahili)")
    ap.add_argument("--target_language", required=True, help="Target language for translation")
    ap.add_argument("--original_query", default=None, help="Original query string, defaults to '{culture} images'")
    args = ap.parse_args()

    if args.original_query is None:
        args.original_query = f"{args.culture} images"

    images_dir = Path(args.images_dir)
    renamed = ensure_sequential_names(images_dir)
    print(f"[INFO] Found {len(renamed)} valid images sequentially named in {images_dir}")

    if args.num_images:
        renamed = renamed[:args.num_images]

    out_data = []
    out_file = Path(args.out_json)
    processed_set = set()
    if out_file.exists():
        try:
            out_data = json.load(open(out_file, "r", encoding="utf-8"))
            processed_set = {Path(r["image_path"]).name for r in out_data}
            print(f"[INFO] Resuming, loaded {len(out_data)} existing records")
        except Exception:
            out_data = []
            processed_set = set()

    prompt = build_prompt(args.target_language)

    for idx, img_path in enumerate(renamed, start=1):
        if img_path.name in processed_set:
            print(f"[{idx}/{len(renamed)}] Skipping {img_path.name}, already processed")
            continue
        print(f"[{idx}/{len(renamed)}] Processing {img_path.name} …")
        parsed, raw_text = send_to_gemini(img_path, prompt)
        if parsed is None:
            safe_sleep()
            continue

        # Prepare raw QA responses
        raw_qa_responses = []
        if "qa_pairs" in parsed:
            for qa in parsed["qa_pairs"]:
                qa_raw = {k: v for k, v in qa.items() if k not in ("translated_question", "translated_answer")}
                raw_qa_responses.append({
                    "type": qa.get("type", "Unknown"),
                    "raw_response": json.dumps(qa_raw, ensure_ascii=False)
                })

        record = {
            "culture": args.culture,
            "image_path": str(img_path),
            "original_query": args.original_query,
            "original_url": "N/A",
            "caption": parsed.get("caption"),
            "translated_caption": parsed.get("translated_caption"),
            "raw_caption_response": raw_text,  # store uncleaned raw text
            "qa_pairs": parsed.get("qa_pairs", []),
            "raw_qa_responses": raw_qa_responses
        }
        out_data.append(record)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote record for {img_path.name} to {out_file}")
        safe_sleep()

    print(f"[INFO] Completed. Total {len(out_data)} records written to {out_file}")

if __name__ == "__main__":
    main()
