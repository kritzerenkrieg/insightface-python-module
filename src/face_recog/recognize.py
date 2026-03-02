from fastapi import APIRouter, UploadFile, File
from face_recog.recognize import recognize_image

router = APIRouter()

@router.post("/recognize")
async def recognize(file: UploadFile = File(...), top_k: int = 1, threshold: float = 0.5):
    image_path = f"./temp/{file.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())
    
    results = recognize_image(image_path, "./db", top_k=top_k, threshold=threshold)
    
    if not results:
        return {"message": "No matches found"}
    
    return [{"name": r['name'], "id": r['id'], "score": r['score']} for r in results]