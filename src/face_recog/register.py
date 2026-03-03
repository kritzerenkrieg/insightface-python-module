import os
import uuid
import base64
from typing import Union
import io
import numpy as np
from PIL import Image
import traceback

# lazy-loaded insightface model
_insight_model = None


def _get_insight_model():
    global _insight_model
    if _insight_model is None:
        try:
            from insightface.app import FaceAnalysis
            _insight_model = FaceAnalysis(allowed_modules=["detection", "recognition"])
            _insight_model.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            print("[register] insightface init failed:")
            traceback.print_exc()
            _insight_model = None
    return _insight_model


def _get_embedding_from_bytes(image_bytes: bytes):
    model = _get_insight_model()
    if model is None:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)
        faces = model.get(arr)
        if not faces:
            return None
        emb = np.array(faces[0].embedding, dtype=np.float32)
        return emb
    except Exception:
        print("[register] embedding extraction failed:")
        traceback.print_exc()
        return None


def _ensure_faces_dir():
    base = os.path.abspath(os.getcwd())
    faces_dir = os.path.join(base, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    return faces_dir


def register_image(image: Union[bytes, str], name: str, db: str = "./db") -> dict:
    """Save an uploaded image (bytes or base64 string) under `faces/<name>/`.

    Returns a dict with `id` and `path`.
    """
    if isinstance(image, str):
        try:
            image_bytes = base64.b64decode(image)
        except Exception:
            raise ValueError("`image` must be base64 string or raw bytes")
    elif isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
    else:
        raise ValueError("Unsupported image type")

    faces_dir = _ensure_faces_dir()
    person_dir = os.path.join(faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    image_id = uuid.uuid4().hex
    filename = f"{image_id}.jpg"
    path = os.path.join(person_dir, filename)

    with open(path, "wb") as f:
        f.write(image_bytes)

    # attempt to compute and save embedding (optional)
    emb = _get_embedding_from_bytes(image_bytes)
    if emb is not None:
        try:
            emb_path = os.path.splitext(path)[0] + ".npy"
            np.save(emb_path, emb)
        except Exception:
            pass

    return {"id": image_id, "path": path, "embedding_saved": emb is not None}