from .utils import load_image, get_embedding_from_image, add_embedding_to_db

def register_image(image_path: str, name: str, db_dir: str = "./db") -> dict:
    img = load_image(image_path)
    emb = get_embedding_from_image(img)
    if emb is None:
        raise ValueError("No face detected or embedding failed")
    eid = add_embedding_to_db(emb, name, db_dir)
    return {"id": eid, "name": name}
