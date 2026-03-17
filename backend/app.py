import streamlit as st
import os
import sys
import time
import base64
import tempfile
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Awwa Stories - Personalized Storybook",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Awwa Stories")
st.subheader("Create your personalized storybook!")

# ── 写真アップロード ──────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a photo of your child",
    type=["jpg", "jpeg", "png"],
    help="Please upload a clear front-facing photo"
)

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)

# ── 性別選択 ──────────────────────────────────────────
gender = st.radio(
    "Select gender",
    options=["girl", "boy"],
    format_func=lambda x: "👧 Girl" if x == "girl" else "👦 Boy",
    horizontal=True
)

# ── 生成ボタン ────────────────────────────────────────
generate_btn = st.button(
    "✨ Generate My Storybook!",
    disabled=uploaded_file is None,
    use_container_width=True,
    type="primary"
)

if generate_btn and uploaded_file:
    # 写真を一時ファイルに保存
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        photo_path = tmp.name

    # 進捗表示
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🎨 Step 1/3: Generating anime face...")
        progress_bar.progress(10)

        from generate import generate_anime_face, generate_char_desc, generate_scenes, OUTPUT_DIR

        # 出力フォルダ
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(OUTPUT_DIR) / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step1: PuLID
        anime_face_path = generate_anime_face(photo_path, out_dir)
        progress_bar.progress(30)

        # Step2: Claude API
        status_text.text("🤖 Step 2/3: Analyzing child's features...")
        char_desc = generate_char_desc(photo_path, gender)
        progress_bar.progress(50)

        # Step3: 全シーン生成
        status_text.text("🖼️ Step 3/3: Generating all scenes... (this takes ~3 minutes)")
        generate_scenes(anime_face_path, char_desc, gender, out_dir)
        progress_bar.progress(90)

        # 結果表示
        status_text.text("✅ Complete!")
        progress_bar.progress(100)

        st.success("🎉 Your personalized storybook is ready!")
        st.balloons()

        # ── 5シーン表示 ──────────────────────────────
        st.markdown("---")
        st.subheader("📖 Your Storybook Pages")

        scene_files = [
            ("Scene 1", out_dir / "scene_01.jpg"),
            ("Scene 2", out_dir / "scene_02.jpg"),
            ("Scene 3", out_dir / "scene_03.jpg"),
            ("Scene 4", out_dir / "scene_04.jpg"),
            ("Scene 5", out_dir / "scene_05.jpg"),
        ]

        for title, path in scene_files:
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_column_width=True)
                st.markdown("")

        # ── PDF出力 ───────────────────────────────────
        st.markdown("---")
        st.subheader("📄 Download as PDF")

        images = []
        for _, path in scene_files:
            if path.exists():
                img = Image.open(path).convert("RGB")
                images.append(img)

        if images:
            pdf_buffer = io.BytesIO()
            images[0].save(
                pdf_buffer,
                format="PDF",
                save_all=True,
                append_images=images[1:]
            )
            pdf_buffer.seek(0)

            st.download_button(
                label="⬇️ Download Storybook PDF",
                data=pdf_buffer,
                file_name="awwa_storybook.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
    finally:
        os.unlink(photo_path)

# ── フッター ──────────────────────────────────────────
st.markdown("---")
st.caption("Powered by Awwa Stories × AI Technology")
