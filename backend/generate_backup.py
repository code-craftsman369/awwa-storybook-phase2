import anthropic, fal_client, base64, urllib.request, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from quality_check import check_image_valid

def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

_load_env()

BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "../outputs/clean_base"
OUTPUT_DIR = BASE_DIR / "../outputs/generated"

def to_b64(path):
    with open(path, "rb") as f:
        ext = str(path).split(".")[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def save_url(url, path):
    urllib.request.urlretrieve(url, path)

# ── Step1: PuLIDでanime_face生成 ──────────────────────
def generate_anime_face(photo_path, out_dir):
    print("🎨 Step1: PuLIDでアニメ顔生成中...")
    result = fal_client.subscribe(
        "fal-ai/pulid",
        arguments={
            "reference_images": [{"image_url": to_b64(photo_path)}],
            "prompt": (
                "children's picture book illustration, cute anime cartoon face close-up, "
                "big bright eyes, wide joyful smile, rosy cheeks, "
                "soft cartoon shading, vibrant colors, chibi style"
            ),
            "negative_prompt": "realistic photo, ugly, deformed, blurry, dark, scary, hijab, headscarf",
            "num_inference_steps": 12,
            "guidance_scale": 1.5,
            "seed": 42,
        },
        with_logs=False,
    )
    anime_face_path = str(out_dir / "anime_face.jpg")
    save_url(result["images"][0]["url"], anime_face_path)
    print(f"✅ anime_face生成完了")
    return anime_face_path

# ── Step2: Claude APIでchar_desc生成 ─────────────────
def generate_char_desc(photo_path, gender):
    print("🤖 Step2: Claude APIで特徴抽出中...")
    client = anthropic.Anthropic()
    with open(photo_path, "rb") as f:
        ext = str(photo_path).split(".")[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        img_data = base64.b64encode(f.read()).decode()

    pronoun = "girl" if gender == "girl" else "boy"
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_data}},
                {"type": "text", "text": (
                    f"Look at this image and describe the appearance of the person in it. "
                    f"Include: hair color, hair style, skin tone, eye color. "
                    f"Format your answer as a single short phrase like this: "
                    f"'[hair style] [hair color] hair, [skin tone] skin, [eye color] eyes, cheerful smile, rosy cheeks' "
                    f"Example: 'straight black hair, light skin, dark brown eyes, cheerful smile, rosy cheeks' "
                    f"Reply with ONLY the description phrase, nothing else."
                )}
            ]
        }]
    )
    char_desc = message.content[0].text.strip()
    print(f"✅ char_desc: {char_desc}")
    return char_desc

# ── flux-kontext（品質チェック+リトライ付き）────────────
RETRY_SEEDS = [42, 99, 15, 7, 1]

def run_kontext_with_retry(bg_path, ref_path, prompt, base_seed, out_path, max_retries=3):
    for i in range(max_retries):
        seed = RETRY_SEEDS[i % len(RETRY_SEEDS)]
        try:
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext",
                arguments={
                    "image_url": to_b64(bg_path),
                    "reference_image_url": to_b64(ref_path),
                    "prompt": prompt,
                    "guidance_scale": 3.5,
                    "num_inference_steps": 28,
                    "seed": seed,
                },
                with_logs=False,
            )
            url = result["images"][0]["url"]
            save_url(url, out_path)

            if check_image_valid(out_path):
                return out_path
            else:
                print(f"  ⚠️ 品質NG seed={seed}, リトライ {i+1}/{max_retries}...")
        except Exception as e:
            print(f"  ⚠️ エラー seed={seed}: {e}, リトライ {i+1}/{max_retries}...")

    print(f"  ❌ {max_retries}回試行しても品質基準未達: {out_path}")
    return out_path

# ── Step3: 全シーン生成 ───────────────────────────────
def generate_scenes(anime_face_path, char_desc, gender, out_dir):
    print("🖼️  Step3: 全シーン生成中...")
    clean = CLEAN_DIR

    scenes_parallel = [
        ("scene_01", str(clean / "scene_01_clean.jpg"),
         f"Add a cute small anime girl sitting on the yellow bulldozer on the RIGHT side. "
         f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. "
         f"She is sitting in the driver seat with a big smile. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         42, str(out_dir / "scene_01.jpg")),

        ("scene_02_left", str(clean / "scene_02_clean.jpg"),
         f"Add a cute small anime girl sitting on the small yellow tractor on the LEFT side. "
         f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. "
         f"She is small, sitting in the driver seat. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         42, str(out_dir / "scene_02_left.jpg")),

        ("scene_03", str(BASE_DIR / "assets/backgrounds/scene_03.jpg"),
         f"Replace the hijab girl on the yellow bulldozer on the LEFT side "
         f"with a cute small anime girl. "
         f"The new girl has {char_desc}. Wearing a purple top. No hijab. No headscarf. "
         f"She is sitting in the driver seat looking forward with a smile. "
         f"Keep the orange cat, forest, dinosaurs, river and everything else exactly the same. "
         f"Match the anime picture book illustration style exactly.",
         42, str(out_dir / "scene_03.jpg")),

        ("scene_04", str(clean / "scene_04_clean.jpg"),
         f"Add a cute small anime girl driving the yellow roller machine on the LEFT side. "
         f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. "
         f"She is sitting in the driver seat with a cheerful smile. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         99, str(out_dir / "scene_04.jpg")),
    ]

    def process(args):
        scene_id, bg_path, prompt, seed, out_path = args
        t = time.time()
        run_kontext_with_retry(bg_path, anime_face_path, prompt, seed, out_path)
        print(f"  ✅ {scene_id} ({time.time()-t:.1f}秒)")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process, args) for args in scenes_parallel]
        for f in as_completed(futures):
            f.result()

    # scene_02右
    print("  scene_02 右キャラ追加中...")
    t = time.time()
    run_kontext_with_retry(
        str(out_dir / "scene_02_left.jpg"), anime_face_path,
        f"Add a cute small chibi anime girl leaning slightly out of the large yellow vehicle on the RIGHT side. "
        f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. "
        f"Her full upper body including arms and torso is visible. Small head proportional to chibi body. "
        f"She is smiling at the cat below. "
        f"Match the anime picture book illustration style exactly. Keep everything else the same.",
        42, str(out_dir / "scene_02.jpg")
    )
    print(f"  ✅ scene_02 ({time.time()-t:.1f}秒)")

    # scene_05 step1
    print("  scene_05 step1中...")
    t = time.time()
    s5_step1 = str(out_dir / "scene_05_step1.jpg")
    run_kontext_with_retry(
        str(clean / "scene_05_clean.jpg"), anime_face_path,
        f"Add a cute small anime girl standing near the barn on the LEFT side. "
        f"The girl has {char_desc}. Wearing a pink long dress. No hijab. No headscarf. "
        f"She is standing holding something small, looking cheerful. "
        f"Keep the father, mother with purple hijab, blue car, barn, chicken and ALL other elements exactly the same. "
        f"Match the anime picture book illustration style exactly.",
        42, s5_step1
    )
    print(f"  ✅ scene_05 step1 ({time.time()-t:.1f}秒)")

    # scene_05 step2
    print("  scene_05 step2中...")
    t = time.time()
    run_kontext_with_retry(
        s5_step1, anime_face_path,
        f"Add a cute small anime girl in the CENTER back seat of the blue car on the RIGHT side. "
        f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. Natural hair. "
        f"She is sitting between the father and mother, smiling happily. "
        f"Do NOT change the bearded father on the left. Do NOT change the mother with purple hijab on the right. "
        f"Match the anime picture book illustration style exactly.",
        7, str(out_dir / "scene_05.jpg")
    )
    print(f"  ✅ scene_05 ({time.time()-t:.1f}秒)")

# ── メイン関数 ────────────────────────────────────────
def generate(photo_path, gender="girl"):
    total_start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 出力フォルダ: {out_dir}")

    anime_face_path = generate_anime_face(photo_path, out_dir)
    char_desc = generate_char_desc(photo_path, gender)
    generate_scenes(anime_face_path, char_desc, gender, out_dir)

    total = (time.time() - total_start) / 60
    print(f"\n🎉 完了! 合計: {total:.1f}分")
    print(f"📁 出力: {out_dir}")
    return str(out_dir)

if __name__ == "__main__":
    generate(
        photo_path="/Users/sunahatatatsuya/awwa-storybook-phase2/test_assets/sample_photo_girl.png",
        gender="girl"
    )
