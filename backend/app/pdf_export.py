from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from pathlib import Path
import json

router = APIRouter()
OUTPUT_DIR = Path("../outputs")
STORY_CONFIG = Path("../assets/stories/story_001/story.json")

@router.get("/api/export-pdf/{session_id}")
def export_pdf(session_id: str):
    # story.jsonからテキスト取得
    with open(STORY_CONFIG) as f:
        story = json.load(f)
    scenes = sorted(story["scenes"], key=lambda x: x["order"])

    buf = BytesIO()
    w, h = landscape(A4)
    c = canvas.Canvas(buf, pagesize=(w, h))

    for scene in scenes:
        img_path = OUTPUT_DIR / f"{session_id}_{scene['scene_id']}.jpg"
        if not img_path.exists():
            continue

        img_area_h = h * 0.78
        c.drawImage(
            ImageReader(str(img_path)),
            0, h - img_area_h,
            width=w, height=img_area_h,
            preserveAspectRatio=True, anchor='c'
        )

        c.setFillColorRGB(1, 1, 1)
        c.rect(0, 0, w, h * 0.22, fill=1, stroke=0)

        c.setFillColorRGB(0.15, 0.15, 0.15)
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(w / 2, h * 0.08, scene.get("text", ""))

        c.showPage()

    c.save()
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=awwa_story_{session_id[:8]}.pdf"}
    )
