from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / ".chroma"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client = None
_embed_fn = None


def _get_client():
    global _client, _embed_fn
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _client, _embed_fn


def retrieve(
    query: str,
    collection: str,
    k: int = 3,
    where: dict | None = None,
) -> list[dict]:
    client, embed_fn = _get_client()
    coll = client.get_collection(collection, embedding_function=embed_fn)
    res = coll.query(query_texts=[query], n_results=k, where=where)
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"text": doc, "meta": meta, "distance": dist})
    return out
