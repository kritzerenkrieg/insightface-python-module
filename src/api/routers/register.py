from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from face_recog.register import register_image

router = APIRouter()


@router.post("/register")
async def register(name: str = Form(...), image: UploadFile = File(...)):
    """Register an image under `faces/<name>/`. Database path is fixed to `./db`.

    - `name` is a form field.
    - `image` is the multipart file field (key: "image").
    """
    try:
        contents = await image.read()
        out = register_image(contents, name)
        return {"message": "Registered", "data": out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))