"""
Step7: Phase2パイプライン テストスクリプト
実行方法:
  cd ~/awwa-storybook-phase2/backend
  source venv/bin/activate
  python test_phase2_step7.py
"""

import sys
import os
import time
from pathlib import Path

# --- パス設定 ---
# このスクリプトをbackend/に置いて実行する想定
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

# 出力・キャッシュディレクトリ確認
OUTPUT_DIR = BACKEND_DIR / "../outputs"
CACHE_DIR = BACKEND_DIR / "../cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- テスト設定 ---
# ★ここを自分の写真パスに変更してください
CHILD_PHOTO_PATH = str(BACKEND_DIR / "../test_assets/girl_sample_photo.jpg")

# テスト対象シーン背景画像
SCENE_02_BG = str(BACKEND_DIR / "assets/backgrounds/scene_02.jpg")

SESSION_ID = "test_phase2_step7"


def check_prerequisites():
    """前提条件チェック"""
    print("=" * 60)
    print("Phase2 Step7 テスト開始")
    print("=" * 60)

    errors = []

    if not Path(CHILD_PHOTO_PATH).exists():
        errors.append(f"❌ 子供写真が見つかりません: {CHILD_PHOTO_PATH}")
    else:
        print(f"✅ 子供写真: {CHILD_PHOTO_PATH}")

    if not Path(SCENE_02_BG).exists():
        errors.append(f"❌ scene_02背景が見つかりません: {SCENE_02_BG}")
    else:
        print(f"✅ scene_02背景: {SCENE_02_BG}")

    # .envチェック
    env_path = BACKEND_DIR / ".env"
    if not env_path.exists():
        errors.append("❌ .envファイルが見つかりません")
    else:
        with open(env_path) as f:
            content = f.read()
        has_fal = "FAL_KEY" in content
        has_claude = "ANTHROPIC_API_KEY" in content
        print(f"✅ .env: FAL_KEY={'あり' if has_fal else '❌なし'}, ANTHROPIC_API_KEY={'あり' if has_claude else '❌なし'}")
        if not has_fal:
            errors.append("❌ .envにFAL_KEYがありません")
        if not has_claude:
            errors.append("❌ .envにANTHROPIC_API_KEYがありません")

    if errors:
        print("\n前提条件エラー:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print("\n前提条件: 全てOK ✅\n")


def test_step1_identity_spec():
    """テスト1: IdentitySpec生成"""
    print("-" * 40)
    print("テスト1: build_identity_spec()")
    print("-" * 40)

    from app.generate import build_identity_spec

    spec = build_identity_spec(CHILD_PHOTO_PATH, SESSION_ID)

    print(f"\n生成されたIdentitySpec:")
    print(f"  gender      : {spec.gender}")
    print(f"  age_band    : {spec.age_band}")
    print(f"  hair_color  : {spec.hair_color}")
    print(f"  hair_length : {spec.hair_length}")
    print(f"  hair_style  : {spec.hair_style}")
    print(f"  eye_color   : {spec.eye_color}")
    print(f"  skin_tone   : {spec.skin_tone}")
    print(f"  description : {spec.full_description}")
    print(f"\n  → prompt fragment: {spec.to_prompt_fragment()}")

    # キャッシュ確認
    cache_path = CACHE_DIR / f"{SESSION_ID}_identity_spec.json"
    assert cache_path.exists(), f"キャッシュが作成されていません: {cache_path}"
    print(f"  → キャッシュ保存: {cache_path} ✅")

    return spec


def test_step2_canonical(spec):
    """テスト2: canonical character生成"""
    print("\n" + "-" * 40)
    print("テスト2: generate_canonical_character()")
    print("-" * 40)

    from app.generate import generate_canonical_character

    canonical_path = generate_canonical_character(CHILD_PHOTO_PATH, spec, SESSION_ID)

    assert Path(canonical_path).exists(), f"canonicalが生成されていません: {canonical_path}"
    size = Path(canonical_path).stat().st_size
    print(f"  → canonical保存: {canonical_path}")
    print(f"  → ファイルサイズ: {size // 1024}KB ✅")

    return canonical_path


def test_step3_scene02(spec):
    """テスト3: scene_02でPhase2生成"""
    print("\n" + "-" * 40)
    print("テスト3: generate_scene(scene_02, use_phase2=True)")
    print("-" * 40)

    from app.generate import generate_scene, _build_prompt_v2

    # プロンプト確認
    prompt = _build_prompt_v2("scene_02", spec)
    print(f"  使用プロンプト:\n  {prompt[:200]}...")

    print(f"\n  scene_02生成開始...")
    start = time.time()

    result_path = generate_scene(
        base_image_path=SCENE_02_BG,
        bboxes=[],
        child_photo_path=CHILD_PHOTO_PATH,
        session_id=SESSION_ID,
        scene_id="scene_02",
        use_phase2=True,
    )

    elapsed = time.time() - start
    assert Path(result_path).exists(), f"出力が生成されていません: {result_path}"
    size = Path(result_path).stat().st_size
    print(f"\n  → 出力: {result_path}")
    print(f"  → ファイルサイズ: {size // 1024}KB")
    print(f"  → 生成時間: {elapsed:.1f}秒 ✅")

    return result_path


def main():
    check_prerequisites()

    try:
        # テスト1: IdentitySpec
        spec = test_step1_identity_spec()

        # テスト2: canonical character
        canonical_path = test_step2_canonical(spec)

        # テスト3: scene_02生成
        result_path = test_step3_scene02(spec)

        print("\n" + "=" * 60)
        print("✅ Step7 全テスト完了!")
        print("=" * 60)
        print(f"\n生成ファイル:")
        print(f"  canonical : {canonical_path}")
        print(f"  scene_02  : {result_path}")
        print(f"\n結果を目視確認してください。")
        print(f"問題があれば generate.py の _build_prompt_v2() を調整します。")

    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
