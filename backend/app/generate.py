import anthropic, fal_client, requests, base64, os, time, json
from dataclasses import dataclass, asdict
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("../outputs")
CACHE_DIR = Path("../cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "character_pack").mkdir(exist_ok=True)
(CACHE_DIR / "scene_outputs").mkdir(exist_ok=True)


@dataclass
class IdentitySpec:
    """子供の外見特徴を保持する安定したデータクラス。全シーンで再利用する。"""
    gender: str
    age_band: str           # "toddler" / "young_child" / "older_child"
    hair_color: str
    hair_length: str        # "short" / "medium" / "long"
    hair_style: str         # "straight" / "wavy" / "curly" / "ponytail"
    eye_color: str
    skin_tone: str          # "light" / "medium" / "tan" / "dark"
    full_description: str   # Claudeが生成した1文の説明（後方互換用）

    def to_prompt_fragment(self) -> str:
        """シーンプロンプトに埋め込む簡潔な特徴文字列を返す"""
        pronoun = "his" if self.gender == "boy" else "her"
        return (
            f"a {self.age_band.replace('_', ' ')} {self.gender} "
            f"with {self.hair_length} {self.hair_style} {self.hair_color} hair, "
            f"{self.eye_color} eyes, {self.skin_tone} skin tone"
        )

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IdentitySpec":
        with open(path) as f:
            return cls(**json.load(f))

def _load_env():
    env = {}
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env

ENV = _load_env()
os.environ["FAL_KEY"] = ENV.get("FAL_KEY", "")
ANTHROPIC_API_KEY = ENV.get("ANTHROPIC_API_KEY", "")


def to_b64(filepath: str, mime: str = "image/jpeg") -> str:
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _is_black(img: Image.Image) -> bool:
    arr = np.array(img.convert("L"), dtype=float)
    return (arr.mean() / 255.0) < 0.02


def analyze_child_photo(child_photo_path: str) -> tuple:
    print("Analyzing child photo with Claude...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with open(child_photo_path, "rb") as f:
        photo_b64 = base64.b64encode(f.read()).decode()
    ext = Path(child_photo_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    analysis = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": photo_b64}},
            {"type": "text", "text": (
                "Describe this child for an anime picture book illustration. "
                "Output exactly two things separated by '|': "
                "1) One sentence: boy or girl, hair color/style, eye color, age "
                "2) Gender word only: 'girl' or 'boy' "
                'Example: "a 5-year-old girl with long dark brown hair and dark brown eyes|girl" '
                "No extra text."
            )}
        ]}]
    )
    raw = analysis.content[0].text.strip().strip('"')
    if '|' in raw:
        desc, gender = raw.split('|', 1)
        desc, gender = desc.strip(), gender.strip()
    else:
        desc = raw
        gender = 'girl' if 'girl' in raw.lower() else 'boy'
    print(f"Child description: {desc}, gender: {gender}")
    return desc, gender


def build_identity_spec(child_photo_path: str, session_id: str) -> IdentitySpec:
    """
    子供の写真を分析してIdentitySpecを生成・キャッシュする。
    2回目以降はキャッシュから読み込む。
    """
    cache_path = CACHE_DIR / f"{session_id}_identity_spec.json"
    if cache_path.exists():
        print(f"  IdentitySpec loaded from cache: {cache_path}")
        return IdentitySpec.load(cache_path)

    print("Analyzing child photo for IdentitySpec...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with open(child_photo_path, "rb") as f:
        photo_b64 = base64.b64encode(f.read()).decode()
    ext = Path(child_photo_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    analysis = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": photo_b64}},
            {"type": "text", "text": (
                "Analyze this child's photo and output a JSON object only. No other text.\n"
                "Required fields:\n"
                '{ "gender": "girl" or "boy",\n'
                '  "age_band": "toddler" or "young_child" or "older_child",\n'
                '  "hair_color": e.g. "dark brown",\n'
                '  "hair_length": "short" or "medium" or "long",\n'
                '  "hair_style": "straight" or "wavy" or "curly" or "ponytail",\n'
                '  "eye_color": e.g. "dark brown",\n'
                '  "skin_tone": "light" or "medium" or "tan" or "dark",\n'
                '  "full_description": one sentence describing the child for anime illustration\n'
                "}\n"
                "Output JSON only."
            )}
        ]}]
    )

    raw = analysis.content[0].text.strip()
    # ```json ... ``` を除去
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  JSON parse failed, using fallback. Raw: {raw[:100]}")
        # フォールバック：旧analyze_child_photoを使う
        desc, gender = analyze_child_photo(child_photo_path)
        data = {
            "gender": gender,
            "age_band": "young_child",
            "hair_color": "dark brown",
            "hair_length": "long",
            "hair_style": "straight",
            "eye_color": "dark brown",
            "skin_tone": "light",
            "full_description": desc,
        }

    spec = IdentitySpec(**data)
    spec.save(cache_path)
    print(f"  IdentitySpec saved: {cache_path}")
    print(f"  → {spec.to_prompt_fragment()}")
    return spec


def generate_canonical_character(
    child_photo_path: str,
    identity_spec: IdentitySpec,
    session_id: str,
) -> str:
    """
    子供の正面アニメキャラ画像（canonical）を1枚生成してキャッシュする。
    これが全シーンのアイデンティティ基準になる。
    """
    canonical_path = str(CACHE_DIR / f"{session_id}_canonical.jpg")
    if os.path.exists(canonical_path):
        print(f"  Canonical loaded from cache: {canonical_path}")
        return canonical_path

    print("Generating canonical character...")
    desc = identity_spec.to_prompt_fragment()
    prompt = (
        f"A soft anime-style picture book illustration. "
        f"A single child character: {desc}. "
        f"The child is smiling gently, facing forward, centered in the frame. "
        f"Simple pastel background. "
        f"Clear face, large expressive eyes, clean lines. "
        f"This is a reference portrait for a children's picture book. "
        f"Do NOT add any text or other characters."
    )
    _run_kontext(child_photo_path, prompt, canonical_path)
    print(f"  Canonical saved: {canonical_path}")
    return canonical_path


def _build_prompt(scene_id: str, child_desc: str, gender: str) -> str:
    pronoun = "his" if gender == "boy" else "her"

    prompts = {
        "scene_01": (
            f"This is an anime-style picture book illustration. "
            f"Find the child character riding a vehicle on the RIGHT half of the image. "
            f"Replace only that character's face and hair to look like: {child_desc}. "
            f"Keep the character's clothing, body, and pose completely unchanged. "
            f"Keep the entire LEFT half and all background elements completely unchanged. "
            f"Do NOT add any text, words, or letters anywhere in the image. "
            f"Preserve the soft anime picture book art style."
        ),
        "scene_02": (
            f"This is an anime-style picture book illustration of a sunny farm with wheat fields. "
            f"There are TWO child characters: a small one on the LEFT on a small yellow tractor, "
            f"and a large one on the RIGHT edge leaning out of a bigger yellow vehicle. "
            f"Change ONLY the face and hair of the large child on the RIGHT "
            f"to look like: {child_desc}. Keep {pronoun} clothing and body unchanged. "
            f"The small child on the LEFT must remain completely unchanged. "
            f"Keep the wheat, trees, orange cat, sky, and all background completely unchanged. "
            f"Do NOT add any text to the image. Preserve the anime picture book art style."
        ),
        "scene_03": (
            f"This is an anime-style picture book illustration of a colorful forest. "
            f"There is a child character sitting on top of a yellow bulldozer in the LEFT half of the image, holding a pink toy. "
            f"An orange cat is also on the bulldozer. "
            f"Replace only the child's face and hair to look like: {child_desc}. "
            f"Keep the child's clothing, body, pose, and the pink toy completely unchanged. "
            f"Keep the orange cat completely unchanged. "
            f"Keep the forest trees, green grass, dinosaurs, river, volcano, and entire RIGHT half completely unchanged. "
            f"Do NOT add any text or words to the image. "
            f"Preserve the anime picture book art style."
        ),
        "scene_04": (
            f"This is an anime-style picture book illustration of a jungle. "
            f"Find the small child character on the LEFT driving a yellow roller machine. "
            f"Replace only that character's face and hair to look like: {child_desc}. "
            f"Keep the character's clothing and body completely unchanged. "
            f"Keep the rocks, blue dinosaur, waterfall, and entire RIGHT half completely unchanged. "
            f"Do NOT add any text or words to the image. "
            f"Preserve the anime picture book art style."
        ),
        "scene_05_left": (
            f"This is a picture book illustration with TWO side-by-side pages. "
            f"The LEFT page shows: a farm scene with a barn, yellow tractor, chickens, and a small standing child. "
            f"The RIGHT page shows: inside a blue car with a bearded man driving, a child in center, woman in purple hijab on right. "
            f"Task: Change ONLY the small standing child on the LEFT farm page "
            f"to look like: {child_desc}. Keep {pronoun} clothing and pose unchanged. "
            f"The RIGHT page with the blue car and all people inside must be pixel-perfect identical to the original. "
            f"The farm background, barn, tractor, chickens, fence on the LEFT must stay unchanged. "
            f"Do NOT add any text to the image. Preserve anime picture book art style."
        ),
        "scene_05_car": (
            f"This illustration has TWO pages side by side. "
            f"LEFT page shows a farm scene. RIGHT page shows inside a blue car. "
            f"The car has: a bearded adult man driving on the LEFT seat, "
            f"a small child in the CENTER seat, and a woman in purple hijab on the RIGHT seat. "
            f"Change ONLY the small child in the CENTER seat "
            f"to look like: {child_desc}. "
            f"The bearded adult male driver on the LEFT must stay completely unchanged. "
            f"The woman in purple hijab on the RIGHT must stay completely unchanged. "
            f"The LEFT page farm scene must stay completely unchanged. "
            f"Do NOT add any text to the image. Preserve anime picture book art style."
        ),
    }
    return prompts.get(scene_id, f"Change the child character to look like: {child_desc}. Keep everything else unchanged. Do not add text.")


def _build_prompt_v2(scene_id: str, identity_spec: IdentitySpec) -> str:
    """
    Phase2版プロンプトビルダー。
    顔の詳細はidentity_specから最小限だけ参照し、
    canonical画像との組み合わせで一貫性を確保する。
    """
    desc = identity_spec.to_prompt_fragment()
    pronoun = "his" if identity_spec.gender == "boy" else "her"

    prompts = {
        "scene_02": (
            f"Anime-style picture book illustration of a sunny farm. "
            f"TWO children: small child on LEFT on yellow tractor, large child on RIGHT in yellow vehicle. "
            f"Update BOTH children's faces and hair to match {desc}. Dark brown eyes. "
            f"The children must have NO headscarf and NO hijab — show natural {identity_spec.hair_color} hair only. "
            f"Keep all clothing, poses, and background completely unchanged. "
            f"Do NOT add text. Anime picture book style."
        ),
        "scene_01": (
            f"Anime-style picture book illustration. "
            f"Find the child on the RIGHT side riding a yellow bulldozer. "
            f"Change ONLY that child's face and hair to match {desc}. "
            f"Eyes must be dark brown. "
            f"The child's face must be bright, cheerful, and well-lit with soft lighting. "
            f"Keep {pronoun} clothing, body, and pose completely unchanged. "
            f"Keep the LEFT side and all backgrounds completely unchanged. "
            f"Preserve anime picture book art style."
        ),
        "scene_03": (
            f"This is an anime-style picture book illustration with TWO side-by-side pages. "
            f"LEFT page: a child sitting on a yellow bulldozer in a forest, holding a pink stuffed animal toy. An orange cat is nearby. "
            f"RIGHT page: green pterodactyl dinosaurs flying over a rocky island with water — do NOT change the RIGHT page at all. "
            f"Task: update ONLY the child's face and hair on the LEFT page to match {desc}. "
            f"Eyes must be dark brown — NOT blue, NOT green. "
            f"Keep {pronoun} existing clothing EXACTLY unchanged — do NOT change clothing color or style. "
            f"Keep {pronoun} body, pose, and the pink toy completely unchanged. "
            f"Keep the orange cat, all trees, and entire RIGHT page completely unchanged. "
            f"Do NOT add any text. Preserve anime picture book art style."
        ),
        "scene_04": (
            f"Anime-style picture book illustration of a jungle. "
            f"The small child on the LEFT driving a yellow roller machine: "
            f"update {pronoun} face and hair to match {desc}. "
            f"Keep {pronoun} clothing and body unchanged. "
            f"Keep rocks, blue dinosaur, waterfall, and entire RIGHT half unchanged. "
            f"Do NOT add any text. Preserve anime picture book art style."
        ),
        "scene_05_left": (
            f"Anime-style picture book illustration. "
            f"TASK: Change ONLY the small child standing near the farm barn. "
            f"Give that child: long dark brown straight hair, light skin, dark brown eyes, no headscarf. "
            f"The child must have a bright cheerful smile and warm happy expression. "
            f"Keep {pronoun} pink clothing and pose unchanged. "
            f"ALL OTHER CHARACTERS AND ELEMENTS MUST BE 100% UNCHANGED: "
            f"The bearded man in the car, the child in the car, the woman in the car — do NOT change their skin color, hair, or faces at all. "
            f"The barn, tractor, chickens, fence, sky — do NOT change. "
            f"Do NOT add text. Preserve anime art style."
        ),
        "scene_05_car": (
            f"Anime-style picture book illustration. "
            f"The farm scene on the left half must stay 100% unchanged — do NOT modify it at all. "
            f"On the right half, inside the blue car: "
            f"Bearded adult man with yellow hat on the LEFT seat — keep 100% unchanged. "
            f"Small child in CENTER seat — give {pronoun} long dark brown straight hair, light skin, dark brown eyes, bright cheerful smile. No headscarf, show only natural hair. Keep {pronoun} pink clothing. "
            f"Adult woman on RIGHT seat — keep 100% unchanged. "
            f"Do NOT add text. Preserve anime picture book style."
        ),
    }
    return prompts.get(
        scene_id,
        f"Update the child character's face and hair to match {desc}. Keep everything else unchanged. Do not add text."
    )


def _run_kontext(
    image_path: str,
    prompt: str,
    out_path: str,
    max_retries: int = 3,
    reference_image_path: str = None,   # Phase2: canonical参照画像（対応確認後に有効化）
    base_image_override: str = None,    # Phase2: 2pass時に背景画像を上書き
) -> str:
    # base_image_overrideが指定された場合はそちらを背景として使う
    actual_image_path = base_image_override if base_image_override else image_path
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  Retry {attempt}/{max_retries-1}...")
            time.sleep(2)
        try:
            # TODO: fal-ai/flux-pro/kontextがimage_url複数対応した場合、
            # reference_image_pathをimage_urlsリストに追加する
            # 現時点ではbase画像のみ使用
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext",
                arguments={
                    "image_url": to_b64(actual_image_path, "image/jpeg"),
                    "prompt": prompt,
                    "guidance_scale": 7.0,
                    "num_inference_steps": 28,
                    "num_images": 1,
                    "safety_tolerance": "5",
                    "seed": 42,
                },
                with_logs=False,
            )
            url = result["images"][0]["url"]
            img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
            if _is_black(img):
                print(f"  Black image on attempt {attempt+1}, retrying...")
                continue
            img.save(out_path, "JPEG", quality=90)
            print(f"  Saved: {out_path}")
            return out_path
        except Exception as e:
            print(f"  Error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
    import shutil
    shutil.copy(image_path, out_path)
    return out_path


def generate_scene(
    base_image_path: str,
    bboxes: list,
    child_photo_path: str,
    session_id: str,
    scene_id: str,
    view_type: str = "front_close",
    patch_map: dict = None,
    child_desc=None,
    gender: str = None,
    use_phase2: bool = True,   # Phase2パイプラインを使うかどうか
) -> str:
    out_path = str(OUTPUT_DIR / f"{session_id}_{scene_id}.jpg")
    if os.path.exists(out_path):
        print(f"Using cached: {out_path}")
        return out_path

    # ── Phase2パイプライン ──────────────────────────────
    if use_phase2:
        print(f"[Phase2] Generating {scene_id}...")

        # 1. IdentitySpec生成（キャッシュあれば再利用）
        identity_spec = build_identity_spec(child_photo_path, session_id)

        # 2. canonical character生成（キャッシュあれば再利用）
        canonical_path = generate_canonical_character(child_photo_path, identity_spec, session_id)

        # 3. scene別プロンプト（v2: canonical参照 + 最小限の顔記述）
        prompt = _build_prompt_v2(scene_id, identity_spec)

        # 4. scene_05のみ2pass処理
        if scene_id == "scene_05":
            tmp = str(OUTPUT_DIR / f"{session_id}_{scene_id}_tmp.jpg")
            print("  scene_05 pass1 (left farm child)...")
            _run_kontext(base_image_path, _build_prompt_v2("scene_05_left", identity_spec), tmp)
            print("  scene_05 pass2 (car center child)...")
            _run_kontext(tmp, _build_prompt_v2("scene_05_car", identity_spec), out_path)
            if os.path.exists(tmp):
                os.remove(tmp)
        else:
            # canonical_pathをreference、base_image_pathを背景として使用
            _run_kontext(base_image_path, prompt, out_path,
                         reference_image_path=canonical_path)

        return out_path

    # ── Phase1パイプライン（フォールバック）─────────────
    if isinstance(child_desc, tuple):
        child_desc, gender = child_desc
    if child_desc is None:
        child_desc, gender = analyze_child_photo(child_photo_path)
    if gender is None:
        gender = "girl"

    if scene_id == "scene_05":
        tmp = str(OUTPUT_DIR / f"{session_id}_{scene_id}_tmp.jpg")
        print("scene_05 pass1 (left page standing child)...")
        _run_kontext(base_image_path, _build_prompt("scene_05_left", child_desc, gender), tmp)
        print("scene_05 pass2 (car center child)...")
        _run_kontext(tmp, _build_prompt("scene_05_car", child_desc, gender), out_path)
        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        print(f"Generating {scene_id}...")
        _run_kontext(base_image_path, _build_prompt(scene_id, child_desc, gender), out_path)

    return out_path
