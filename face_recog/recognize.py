import numpy as np
from .utils import load_image, get_embedding_from_image, load_all_embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def recognize_image(image_path: str, db_dir: str = "./db", top_k: int = 1, threshold: float = 0.5) -> list:
    img = load_image(image_path)
    emb = get_embedding_from_image(img)
    if emb is None:
        return []
    db_embs, names, ids = load_all_embeddings(db_dir)
    if db_embs.size == 0:
        return []
    # compute similarities
    sims = [cosine_similarity(emb, db_embs[i]) for i in range(db_embs.shape[0])]
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        score = sims[int(i)]
        if score >= threshold:
            results.append({"id": ids[int(i)], "name": names[int(i)], "score": float(score)})
    return results
