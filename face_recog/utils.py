import os
import uuid
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis

_ANALYZER = None

def get_analyzer(det_size=(640, 640)):
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        _ANALYZER.prepare(ctx_id=-1, det_size=det_size)
    return _ANALYZER

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_embedding_from_image(img):
    fa = get_analyzer()
    faces = fa.get(img)
    if not faces:
        return None
    # choose largest face by bbox area
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = best.embedding
    if emb is None:
        return None
    return np.array(emb, dtype=np.float32)

def ensure_db(db_dir="./db"):
    os.makedirs(db_dir, exist_ok=True)
    entries_path = os.path.join(db_dir, "entries.json")
    if not os.path.exists(entries_path):
        with open(entries_path, "w") as f:
            json.dump([], f)
    return entries_path

def load_entries(db_dir="./db"):
    entries_path = ensure_db(db_dir)
    with open(entries_path, "r") as f:
        return json.load(f)

def save_entries(entries, db_dir="./db"):
    entries_path = ensure_db(db_dir)
    with open(entries_path, "w") as f:
        json.dump(entries, f, indent=2)

def add_embedding_to_db(embedding, name, db_dir="./db"):
    ensure_db(db_dir)
    eid = uuid.uuid4().hex
    emb_path = os.path.join(db_dir, f"emb_{eid}.npy")
    np.save(emb_path, embedding)
    entries = load_entries(db_dir)
    entries.append({"id": eid, "name": name, "emb_path": os.path.basename(emb_path)})
    save_entries(entries, db_dir)
    return eid

def load_all_embeddings(db_dir="./db"):
    entries = load_entries(db_dir)
    embeddings = []
    names = []
    ids = []
    for e in entries:
        emb_f = os.path.join(db_dir, e["emb_path"])
        if os.path.exists(emb_f):
            emb = np.load(emb_f)
            embeddings.append(emb)
            names.append(e.get("name"))
            ids.append(e.get("id"))
    if embeddings:
        return np.vstack(embeddings), names, ids
    return np.array([]), [], []
