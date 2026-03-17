from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, uuid, shutil
from pathlib import Path
from app.story import router as story_router
from app.pdf_export import router as pdf_router

app = FastAPI(title="Awwa Storybook API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../temp")
OUTPUT_DIR = Path("../outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="../outputs"), name="outputs")
app.include_router(story_router)
app.include_router(pdf_router)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_photo(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "JPG or PNG only")
    session_id = str(uuid.uuid4())
    ext = "jpg" if file.content_type == "image/jpeg" else "png"
    save_path = UPLOAD_DIR / f"{session_id}.{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"session_id": session_id, "filename": f"{session_id}.{ext}"}

@app.delete("/api/cleanup/{session_id}")
def cleanup(session_id: str):
    for f in UPLOAD_DIR.glob(f"{session_id}.*"):
        f.unlink()
    for f in OUTPUT_DIR.glob(f"{session_id}_*"):
        f.unlink()
    return {"status": "cleaned"}
