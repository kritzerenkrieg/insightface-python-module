Modular InsightFace Register & Recognize (image-based)

This small modular system provides two main capabilities:
- Register a person by image input (extracts face embedding and stores it)
- Recognize a person from an image against the stored database

Setup

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Usage

Register:

```bash
python cli.py register --image /path/to/person.jpg --name "Alice"
```

Recognize:

```bash
python cli.py recognize --image /path/to/query.jpg
```

Database

By default the DB is stored in `./db` as individual `.npy` embeddings and an `entries.json` index.

Notes

- This is image-only (not real-time). It uses `insightface`'s `FaceAnalysis` for detection + embeddings.
- You may need to download model weights on first run; the library will do that automatically.