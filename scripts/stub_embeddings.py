"""Tiny OpenAI-compatible /v1/embeddings server for tests. Not a real model.

Vectors are bag-of-words hashes, so texts sharing words score as similar. That's
enough to exercise the pgvector code path (embed on put, similarity search) in CI
without an API key.

    uv run python scripts/stub_embeddings.py 18081
"""

import hashlib
import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIMS = 64


def embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        out = {
            "object": "list",
            "model": body.get("model", "stub"),
            "data": [{"object": "embedding", "index": i, "embedding": embed(str(t))} for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 18081), Handler).serve_forever()
