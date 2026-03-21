import anthropic, fal_client, base64, urllib.request, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw
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
OUTPUT_DIR = BASE_DIR / "outputs/generated"

CLEAN_BASE_CONFIG = {
    "girl": {
        "dir": BASE_DIR / "outputs/clean_base/girl",
        "scenes": {
            "scene_01": [{"cx": 0.715, "cy": 0.385, "rw": 0.060, "rh": 0.14, "fill": (200, 190, 170)}],
            "scene_02": [{"pixel": [943, 603, 1227, 1073], "fill": (178, 166, 136)}],
            "scene_04": [{"cx": 0.210, "cy": 0.29, "rw": 0.040, "rh": 0.15, "fill": (195, 185, 165)}],
            "scene_05": [
                {"cx": 0.240, "cy": 0.54, "rw": 0.0375, "rh": 0.14, "fill": (195, 185, 165)},
                {"cx": 0.760, "cy": 0.55, "rw": 0.055, "rh": 0.175, "fill": (210, 195, 175)},
            ],
        },
        "bg_dir": BASE_DIR / "assets/backgrounds/girl",
    },
    "boy": {
        "dir": BASE_DIR / "outputs/clean_base/boy",
        "scenes": {
            "scene_01": [{"cx": 0.195, "cy": 0.385, "rw": 0.052, "rh": 0.125, "fill": (180, 195, 210)}],
            "scene_02": [{"cx": 0.205, "cy": 0.345, "rw": 0.040, "rh": 0.115, "fill": (195, 185, 165)}],
            "scene_04": [{"cx": 0.210, "cy": 0.265, "rw": 0.035, "rh": 0.120, "fill": (195, 185, 165)}],
            "scene_05": [
                {"cx": 0.240, "cy": 0.510, "rw": 0.038, "rh": 0.130, "fill": (195, 185, 165)},
                {"cx": 0.755, "cy": 0.500, "rw": 0.048, "rh": 0.150, "fill": (248, 211, 182)},
            ],
        },
        "bg_dir": BASE_DIR / "assets/backgrounds/boy",
    },
}

def create_clean_base(gender):
    cfg = CLEAN_BASE_CONFIG[gender]
    out_dir = cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_dir = cfg["bg_dir"]

    def paint(img, ellipses, w, h):
        draw = ImageDraw.Draw(img)
        for e in ellipses:
            if "pixel" in e:
                draw.ellipse(e["pixel"], fill=e["fill"])
            else:
                cx_px = int(e["cx"] * w)
                cy_px = int(e["cy"] * h)
                rw_px = int(e["rw"] * w)
                rh_px = int(e["rh"] * h)
                draw.ellipse([cx_px-rw_px, cy_px-rh_px, cx_px+rw_px, cy_px+rh_px], fill=e["fill"])
        return img

    for scene_key, ellipses in cfg["scenes"].items():
        src = bg_dir / f"{scene_key}.jpg"
        if not src.exists():
            print(f"  ⚠️ {src} が見つかりません")
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        img = paint(img, ellipses, w, h)
        out_path = out_dir / f"{scene_key}_clean.jpg"
        img.save(out_path, quality=95)
        print(f"  ✅ {out_path.name}")

    print(f"clean_base生成完了: {out_dir}")

def to_b64(path):
    with open(path, "rb") as f:
        ext = str(path).split(".")[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def save_url(url, path):
    urllib.request.urlretrieve(url, path)

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
                    f"Look at this child's photo. Describe the {pronoun}'s physical appearance "
                    f"in a short phrase for an anime illustration prompt. "
                    f"Include: hair color, hair style, skin tone, eye color. "
                    f"Format: '[hair style] [hair color] hair, [skin tone] skin, [eye color] eyes, cheerful smile, rosy cheeks' "
                    f"Example: 'straight black hair, light skin, dark brown eyes, cheerful smile, rosy cheeks' "
                    f"Reply with ONLY the description phrase, nothing else."
                )}
            ]
        }]
    )
    char_desc = message.content[0].text.strip()
    print(f"✅ char_desc: {char_desc}")
    return char_desc

RETRY_SEEDS = [42, 99, 15, 7, 1]

def run_kontext_with_retry(bg_path, ref_path, prompt, base_seed, out_path, max_retries=3, guidance_scale=3.5):
    for i in range(max_retries):
        seed = base_seed if i == 0 else RETRY_SEEDS[i % len(RETRY_SEEDS)]
        try:
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext",
                arguments={
                    "image_url": to_b64(bg_path),
                    "reference_image_url": to_b64(ref_path),
                    "prompt": prompt,
                    "guidance_scale": guidance_scale,
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

def generate_scenes(anime_face_path, char_desc, gender, out_dir):
    print(f"🖼️  Step3: 全シーン生成中 (gender={gender})...")
    clean = CLEAN_BASE_CONFIG[gender]["dir"]
    bg_dir = CLEAN_BASE_CONFIG[gender]["bg_dir"]

    if gender == "girl":
        _generate_scenes_girl(anime_face_path, char_desc, clean, bg_dir, out_dir)
    else:
        _generate_scenes_boy(anime_face_path, char_desc, clean, bg_dir, out_dir)

def _run_parallel(scenes, anime_face_path):
    def process(args):
        scene_id, bg_path, prompt, seed, out_path = args
        t = time.time()
        run_kontext_with_retry(bg_path, anime_face_path, prompt, seed, out_path)
        print(f"  ✅ {scene_id} ({time.time()-t:.1f}秒)")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process, args) for args in scenes]
        for f in as_completed(futures):
            f.result()

def _generate_scenes_girl(anime_face_path, char_desc, clean, bg_dir, out_dir):
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

        ("scene_03", str(bg_dir / "scene_03.jpg"),
         f"Replace the hijab girl on the yellow bulldozer on the LEFT side with a cute small anime girl. "
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
         42, str(out_dir / "scene_04.jpg")),
    ]
    _run_parallel(scenes_parallel, anime_face_path)

    print("  scene_02 右キャラ追加中...")
    t = time.time()
    run_kontext_with_retry(
        str(out_dir / "scene_02_left.jpg"), anime_face_path,
        f"Add a cute small chibi anime girl leaning slightly out of the large yellow vehicle on the RIGHT side. "
        f"The girl has {char_desc}. Wearing a pink top. No hijab. No headscarf. "
        f"Her full upper body including arms and torso is visible. Small head proportional to chibi body. "
        f"She is smiling at the cat below. "
        f"Match the anime picture book illustration style exactly. Keep everything else the same.",
        42, str(out_dir / "scene_02.jpg"))
    print(f"  ✅ scene_02 ({time.time()-t:.1f}秒)")

    print("  scene_05 step1中...")
    t = time.time()
    s5_step1 = str(out_dir / "scene_05_step1.jpg")
    run_kontext_with_retry(
        str(clean / "scene_05_clean.jpg"), anime_face_path,
        f"Add a cute small anime girl standing near the barn on the LEFT side. "
        f"The girl has {char_desc}. Wearing a pink long dress. No hijab. No headscarf. "
        f"She is standing holding something small, looking toward the RIGHT with a cheerful smile. "
        f"Keep the father, mother with purple hijab, blue car, barn, chicken and ALL other elements exactly the same. "
        f"Match the anime picture book illustration style exactly.",
        42, s5_step1)
    print(f"  ✅ scene_05 step1 ({time.time()-t:.1f}秒)")

    print("  scene_05 step2中...")
    t = time.time()
    run_kontext_with_retry(
        s5_step1, anime_face_path,
        f"Replace the PINK OVAL shape in the center back seat with a cute small anime GIRL. "
        f"The girl has {char_desc}. Wearing a PINK top. Long hair. No glasses. No headscarf. No hijab. "
        f"She is sitting between the father and mother, smiling happily. "
        f"The pink oval must be completely replaced by the girl's face and body. "
        f"Do NOT change the bearded father on the left. Do NOT change the mother on the right. "
        f"Match the anime picture book illustration style exactly.",
        15, str(out_dir / "scene_05.jpg"))
    print(f"  ✅ scene_05 ({time.time()-t:.1f}秒)")

def _generate_scenes_boy(anime_face_path, char_desc, clean, bg_dir, out_dir):
    scenes_parallel = [
        ("scene_01", str(clean / "scene_01_clean.jpg"),
         f"Add a cute small anime boy sitting on the yellow bulldozer on the LEFT side. "
         f"The boy has {char_desc}. Wearing a blue top. "
         f"He is sitting in the driver seat with a big smile. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         42, str(out_dir / "scene_01.jpg")),

        ("scene_02_left", str(clean / "scene_02_clean.jpg"),
         f"Add a cute small anime boy sitting on the small yellow tractor on the LEFT side. "
         f"The boy has {char_desc}. Wearing a blue top. "
         f"He is small, sitting in the driver seat. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         42, str(out_dir / "scene_02_left.jpg")),

        ("scene_03", str(bg_dir / "scene_03.jpg"),
         f"Replace the boy on the yellow bulldozer on the LEFT side with a cute small anime boy. "
         f"The new boy has {char_desc}. Wearing a blue top. "
         f"He is sitting in the driver seat looking forward with a smile, holding a white rabbit toy. "
         f"Keep the orange cat, forest, dinosaurs, river and everything else exactly the same. "
         f"Match the anime picture book illustration style exactly.",
         42, str(out_dir / "scene_03.jpg")),

        ("scene_04", str(clean / "scene_04_clean.jpg"),
         f"Add a cute small anime boy driving the yellow roller machine on the LEFT side. "
         f"The boy has {char_desc}. Wearing a blue top. "
         f"He is sitting in the driver seat with a cheerful smile. "
         f"Match the anime picture book illustration style exactly. Keep everything else the same.",
         99, str(out_dir / "scene_04.jpg")),
    ]
    _run_parallel(scenes_parallel, anime_face_path)

    print("  scene_02 右キャラ追加中...")
    t = time.time()
    run_kontext_with_retry(
        str(out_dir / "scene_02_left.jpg"), anime_face_path,
        f"Add a cute small chibi anime boy leaning slightly out of the large yellow vehicle on the RIGHT side. "
        f"The boy has {char_desc}. Wearing a blue top. "
        f"His full upper body including arms and torso is visible. Small head proportional to chibi body. "
        f"He is smiling at the cat below. "
        f"Match the anime picture book illustration style exactly. Keep everything else the same.",
        42, str(out_dir / "scene_02.jpg"))
    print(f"  ✅ scene_02 ({time.time()-t:.1f}秒)")

    # scene_05: 女の子と完全同じ2ステップ方式
    print("  scene_05 step1中...")
    t = time.time()
    s5_step1 = str(out_dir / "scene_05_step1.jpg")
    run_kontext_with_retry(
        str(clean / "scene_05_clean.jpg"), anime_face_path,
        f"Add a cute small anime boy standing near the barn on the LEFT side. "
        f"The boy has {char_desc}. Wearing a blue top and dark pants. "
        f"He is standing holding a white rabbit toy, looking toward the RIGHT with a cheerful smile. "
        f"No hijab. No headscarf. "
        f"Keep barn, chicken and ALL other elements exactly the same. "
        f"Match the anime picture book illustration style exactly.",
        42, s5_step1)
    print(f"  ✅ scene_05 step1 ({time.time()-t:.1f}秒)")

    print("  scene_05 step2中...")
    t = time.time()
    run_kontext_with_retry(
        s5_step1, anime_face_path,
        f"Replace the PINK OVAL shape with a cute small anime boy's face and upper body. "
        f"The boy has {char_desc}. Wearing a BLUE top. "
        f"Keep EVERYTHING else completely unchanged - the father, the mother, the car interior, the background. "
        f"Only the pink oval area should change. Nothing else.",
        42, str(out_dir / "scene_05.jpg"))
    print(f"  ✅ scene_05 ({time.time()-t:.1f}秒)")



def generate(photo_path, gender="girl"):
    total_start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"{gender}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 出力フォルダ: {out_dir}")

    clean_dir = CLEAN_BASE_CONFIG[gender]["dir"]
    if not (clean_dir / "scene_01_clean.jpg").exists():
        print("🔧 clean_base生成中...")
        create_clean_base(gender)

    anime_face_path = generate_anime_face(photo_path, out_dir)
    char_desc = generate_char_desc(photo_path, gender)
    generate_scenes(anime_face_path, char_desc, gender, out_dir)

    total = (time.time() - total_start) / 60
    print(f"\n🎉 完了! 合計: {total:.1f}分")
    print(f"📁 出力: {out_dir}")
    return str(out_dir)

if __name__ == "__main__":
    import sys
    gender = sys.argv[1] if len(sys.argv) > 1 else "girl"
    photo = sys.argv[2] if len(sys.argv) > 2 else \
        "/Users/sunahatatatsuya/awwa-storybook-phase2/test_assets/sample_photo_girl.png"
    generate(photo_path=photo, gender=gender)
