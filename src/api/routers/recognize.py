from fastapi import APIRouter, UploadFile, File, HTTPException
from face_recog.recognize import recognize_image

router = APIRouter()

@router.post("/recognize")
async def recognize(file: UploadFile = File(...), db: str = "./db", top_k: int = 1, threshold: float = 0.5):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        contents = await file.read()
        # Save the file temporarily for recognition
        with open("temp_image.jpg", "wb") as f:
            f.write(contents)

        results = recognize_image("temp_image.jpg", db, top_k=top_k, threshold=threshold)
        
        if not results:
            return {"message": "No matches found"}
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))