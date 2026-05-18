from pathlib import Path
import re
import yaml
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent.parent
KB_DIR = Path(__file__).parent / "domain_kb"
CORPUS_DIR = ROOT / "corpus"
CHROMA_DIR = ROOT / ".chroma"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_markdown_by_h2(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part or not part.startswith("##"):
            continue
        first_line, _, _ = part.partition("\n")
        title = first_line.lstrip("#").strip()
        chunks.append((title, part))
    return chunks


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embed_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def build_domain_kb() -> int:
    client = _client()
    embed_fn = _embed_fn()
    try:
        client.delete_collection("domain_kb")
    except Exception:
        pass
    coll = client.create_collection("domain_kb", embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for md_path in sorted(KB_DIR.glob("*.md")):
        domain = md_path.stem
        for i, (title, body) in enumerate(chunk_markdown_by_h2(md_path.read_text())):
            ids.append(f"{domain}::{i}")
            docs.append(body)
            metas.append({"domain": domain, "title": title, "source": md_path.name})

    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def build_user_corpus() -> int:
    client = _client()
    embed_fn = _embed_fn()
    try:
        client.delete_collection("user_corpus")
    except Exception:
        pass
    coll = client.create_collection("user_corpus", embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for md_path in sorted(CORPUS_DIR.glob("*.md")):
        for i, (title, body) in enumerate(chunk_markdown_by_h2(md_path.read_text())):
            ids.append(f"{md_path.stem}::{i}")
            docs.append(body)
            metas.append({"title": title, "source": md_path.name})

    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def load_yaml_kb() -> dict:
    out = {}
    for yaml_path in sorted(KB_DIR.glob("*.yaml")):
        out[yaml_path.stem] = yaml.safe_load(yaml_path.read_text())
    return out


if __name__ == "__main__":
    nd = build_domain_kb()
    nc = build_user_corpus()
    print(f"Indexed {nd} domain chunks and {nc} corpus chunks")
