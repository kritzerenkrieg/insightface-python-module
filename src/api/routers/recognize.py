from fastapi import APIRouter, UploadFile, File, HTTPException
from face_recog.recognize import recognize_image

router = APIRouter()


@router.post("/recognize")
async def recognize(image: UploadFile = File(...), top_k: int = 1, threshold: float = 0.5):
    """Recognize faces in uploaded image.

    - multipart file field key must be `image` (consistent with `/register`).
    - `db` is fixed to `./db` and not configurable via the API.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await image.read()
        # Save the file temporarily for recognition
        with open("temp_image.jpg", "wb") as f:
            f.write(contents)

        results = recognize_image("temp_image.jpg", top_k=top_k, threshold=threshold)

        if not results:
            return {"message": "No matches found"}

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))