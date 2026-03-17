import json, concurrent.futures
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.generate import generate_scene, analyze_child_photo

router = APIRouter()
STORY_CONFIG = Path("../assets/stories/story_001/story.json")
OUTPUT_DIR = Path("../outputs")
generation_status = {}


def load_story():
    with open(STORY_CONFIG) as f:
        return json.load(f)


def _generate_one(scene, session_id, child_photo_path, child_desc, gender):
    try:
        generate_scene(
            base_image_path=scene["base_image"],
            bboxes=scene["face_bboxes"],
            child_photo_path=child_photo_path,
            session_id=session_id,
            scene_id=scene["scene_id"],
            view_type=scene.get("view_type", "front_close"),
            child_desc=child_desc,
            gender=gender,
        )
        return {
            "scene_id": scene["scene_id"],
            "order": scene["order"],
            "text": scene["text"],
            "image_url": f"/outputs/{session_id}_{scene['scene_id']}.jpg",
            "success": True,
        }
    except Exception as e:
        print(f"Error scene {scene['scene_id']}: {e}")
        return {"scene_id": scene["scene_id"], "order": scene["order"],
                "text": scene.get("text", ""), "image_url": None, "success": False, "error": str(e)}


def run_generation(session_id: str, child_photo_path: str):
    story = load_story()
    scenes = sorted(story["scenes"], key=lambda x: x["order"])
    total = len(scenes)

    generation_status[session_id].update({
        "status": "analyzing", "progress": 0, "total": total, "outputs": []
    })

    try:
        child_desc, gender = analyze_child_photo(child_photo_path)
        generation_status[session_id]["child_desc"] = child_desc
        generation_status[session_id]["gender"] = gender
    except Exception as e:
        generation_status[session_id].update({"status": "error", "error": f"Photo analysis failed: {e}"})
        return

    generation_status[session_id]["status"] = "running"

    # scene_02と05は2パスなので並列数を抑える
    results = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {
            executor.submit(_generate_one, scene, session_id, child_photo_path, child_desc, gender): i
            for i, scene in enumerate(scenes)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            generation_status[session_id]["progress"] += 1
            print(f"Progress: {generation_status[session_id]['progress']}/{total}")

    generation_status[session_id]["outputs"] = [r for r in results if r and r.get("success")]
    generation_status[session_id]["status"] = "completed"


class GenerateRequest(BaseModel):
    session_id: str
    filename: str


@router.post("/api/generate")
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    if req.session_id in generation_status and \
       generation_status[req.session_id]["status"] in ["running", "analyzing", "completed"]:
        return {"status": "already_started", "session_id": req.session_id}
    generation_status[req.session_id] = {
        "status": "running", "progress": 0, "total": 5, "outputs": []
    }
    photo_path = f"../temp/{req.filename}"
    background_tasks.add_task(run_generation, req.session_id, photo_path)
    return {"status": "started", "session_id": req.session_id}


@router.get("/api/status/{session_id}")
def get_status(session_id: str):
    if session_id not in generation_status:
        return JSONResponse(status_code=404, content={"error": "session not found"})
    return generation_status[session_id]
