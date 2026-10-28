"""Face database stored as JSONL: one {"name": ..., "embedding": [...]} per line.
Same format host_server.py reads."""
import json
import os

import numpy as np

JSONL_PATH = "face_embeddings.jsonl"


def save_face_json(name, embedding, path=JSONL_PATH):
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "embedding": emb.tolist()}) + "\n")


def load_face_db_json(path=JSONL_PATH):
    db = {}
    if not os.path.exists(path):
        return db
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("name") and isinstance(obj.get("embedding"), list):
                db[obj["name"]] = np.asarray(obj["embedding"], dtype=np.float32)
    return db
