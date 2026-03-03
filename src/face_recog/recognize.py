import os
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import traceback

# lazy insightface model for embeddings
_insight_model = None


def _get_insight_model():
    global _insight_model
    if _insight_model is None:
        try:
            from insightface.app import FaceAnalysis
            _insight_model = FaceAnalysis(allowed_modules=["detection", "recognition"])
            _insight_model.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            print("[recognize] insightface init failed:")
            traceback.print_exc()
            _insight_model = None
    return _insight_model


def _get_embedding_from_path(image_path: str):
    """Return embedding numpy array or None."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception:
        return None

    model = _get_insight_model()
    if model is None:
        return None

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img)
        faces = model.get(arr)
        if not faces:
            return None
        emb = np.array(faces[0].embedding, dtype=np.float32)
        return emb
    except Exception:
        print("[recognize] embedding extraction failed:")
        traceback.print_exc()
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def recognize_image(image_path: str, db: str = "./db", top_k: int = 1, threshold: float = 0.5) -> List[Dict]:
    """Placeholder recognizer.

    For now this function returns an empty list (no matches).
    Replace with actual model-based recognition logic when available.
    """
    # Basic sanity checks
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Compute embedding for the query image. If embedding can't be computed,
    # return no matches (we no longer fall back to exact file matching).
    query_emb = _get_embedding_from_path(image_path)
    if query_emb is None:
        return []

    faces_dir = os.path.join(os.getcwd(), "faces")
    if not os.path.isdir(faces_dir):
        return []

    candidates = []
    # Walk through saved faces and load .npy embeddings
    for root, _, files in os.walk(faces_dir):
        for fname in files:
            if not fname.lower().endswith('.npy'):
                continue
            fpath = os.path.join(root, fname)
            try:
                emb = np.load(fpath)
            except Exception:
                continue
            person = os.path.basename(root)
            cid = os.path.splitext(fname)[0]
            score = _cosine_similarity(query_emb, emb)
            candidates.append({"name": person, "id": cid, "score": score})

    # sort by score descending and filter by threshold
    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    filtered = [c for c in candidates if c.get('score', 0) >= threshold]
    return filtered[:top_k]