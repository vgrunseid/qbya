#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed.py — Indexar chunks (JSONL de 2-transform.py) en Chroma con embeddings.

Compatibilidad con el formato actual de 2-transform.py:
{
  "id": "DOCID#0000",
  "text": "...",
  "metadata": {
    "doc_id": "...",
    "doc_name": "archivo.txt",
    "drug_name": "alplax xr",
    "section_canonical": "INDICACIONES",
    "section_raw": "Indicaciones",
    "section_confidence": 0.92,
    "chunk_index": 0,
    "char_start": 1234,
    "char_end": 2345,
    "source_path": "/ruta/archivo.txt"
  }
}

Uso:
  # E5 local
  python embed.py -f out_chunks/chunks.jsonl -p chroma/prospectos -c prospectos \
    --provider e5 --e5-model intfloat/multilingual-e5-base --batch-size 128 --skip-existing

  # OpenAI
  export OPENAI_API_KEY="..."
  python embed.py -f out_chunks/chunks.jsonl -p chroma/prospectos -c prospectos \
    --provider openai --openai-model text-embedding-3-small --batch-size 256 --skip-existing

Notas:
- Por defecto compone el texto a embeder como:
    "[drug_name] · [section_canonical] — [text]"
  Cambiá con --compose plain para usar solo el "text" crudo del chunk.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

# Vector store
from langchain_community.vectorstores import Chroma

# ---------- Protocolo de embeddings ----------
class EmbeddingFn(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

# ---------- utilidades JSONL ----------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] línea {ln} inválida: {e}")

# ---------- proveedores de embeddings ----------
def make_openai_embeddings(model_name: str) -> EmbeddingFn:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)

class E5Embeddings:
    """
    Wrapper simple para modelos E5 (intfloat/*) con Sentence-Transformers.
    Aplica el prefijo 'passage: ' a documentos y 'query: ' a consultas.
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
        return vecs

    def embed_query(self, text: str) -> List[float]:
        q = f"query: {text}"
        vec = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        return vec

def make_e5_embeddings(model_name: str) -> EmbeddingFn:
    return E5Embeddings(model_name=model_name)

# ---------- helpers Chroma ----------
def get_or_create_chroma(collection: str, persist_dir: Path, emb: EmbeddingFn) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        collection_name=collection,
        embedding_function=emb,
        persist_directory=str(persist_dir),
    )
    return vs

def existing_ids_in_chroma(vs: Chroma, ids: List[str]) -> set:
    try:
        col = vs._collection  # acceso interno de Chroma
        res = col.get(ids=ids, limit=len(ids))
        return set(res.get("ids", []))
    except Exception:
        return set()

# ---------- composición del texto a embeder ----------
def compose_text_for_embedding(text: str, md: Dict, mode: str = "with_meta") -> str:
    """
    mode:
      - "with_meta" (default): "[drug] · [section] — text"
      - "plain": solo text
    """
    if mode == "plain":
        return (text or "").strip()

    drug = (md.get("drug_name") or "").strip()
    sec = (md.get("section_canonical") or md.get("section_raw") or "").strip()
    pieces = []
    if drug:
        pieces.append(drug)
    if sec:
        pieces.append(sec)
    prefix = " · ".join(pieces)
    if prefix:
        return f"{prefix} — {text}".strip()
    return (text or "").strip()

# ---------- pipeline principal ----------
def index_jsonl_to_chroma(jsonl_path: Path,
                          persist_dir: Path,
                          collection: str,
                          provider: str,
                          openai_model: str,
                          e5_model: str,
                          batch_size: int,
                          skip_existing: bool,
                          compose_mode: str) -> None:
    # Embeddings
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Falta OPENAI_API_KEY en el entorno (provider=openai).")
        emb = make_openai_embeddings(openai_model)
        print(f"Embeddings: OpenAI ({openai_model})")
    elif provider == "e5":
        emb = make_e5_embeddings(e5_model)
        print(f"Embeddings: E5 local ({e5_model})")
    else:
        raise ValueError(f"Provider no soportado: {provider}")

    # Vector store
    vs = get_or_create_chroma(collection=collection, persist_dir=persist_dir, emb=emb)
    print(f"Chroma persist_dir: {persist_dir} | collection: {collection}")

    # Batching
    texts: List[str] = []
    metas: List[Dict] = []
    ids:   List[str] = []
    total_in = 0
    total_added = 0

    def flush_batch():
        nonlocal texts, metas, ids, total_added
        if not texts:
            return

        add_ids = ids
        add_texts = texts
        add_metas = metas

        if skip_existing:
            already = existing_ids_in_chroma(vs, add_ids)
            if already:
                filt_texts, filt_metas, filt_ids = [], [], []
                for t, m, i in zip(add_texts, add_metas, add_ids):
                    if i not in already:
                        filt_texts.append(t); filt_metas.append(m); filt_ids.append(i)
                add_texts, add_metas, add_ids = filt_texts, filt_metas, filt_ids
                if not add_ids:
                    texts, metas, ids = [], [], []
                    return

        vs.add_texts(texts=add_texts, metadatas=add_metas, ids=add_ids)
        vs.persist()
        total_added += len(add_ids)
        texts, metas, ids = [], [], []

    # Leer JSONL
    for rec in iter_jsonl(jsonl_path):
        total_in += 1
        raw_text = rec.get("text", "")
        md = rec.get("metadata", {}) or {}
        _id = rec.get("id") or f"row-{total_in:09d}"

        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        embed_text = compose_text_for_embedding(raw_text, md, mode=compose_mode)

        texts.append(embed_text)
        metas.append(md)
        ids.append(_id)

        if len(texts) >= batch_size:
            flush_batch()

    # último flush
    flush_batch()

    print(f"Listo: recibidos {total_in} registros, indexados {total_added}.")
    print("Tip: para probar, usá tu 4-retrieve.py con filtros por drug/section.")

# ---------- CLI ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Construir embeddings y persistir en Chroma desde un JSONL de chunks.")
    ap.add_argument("-f", "--file", required=True, help="Ruta a chunks.jsonl")
    ap.add_argument("-p", "--persist-dir", default=str(Path("chroma") / "prospectos"),
                    help="Carpeta de persistencia de Chroma (default: ./chroma/prospectos)")
    ap.add_argument("-c", "--collection", default="prospectos", help="Nombre de la colección (default: prospectos)")
    ap.add_argument("--provider", choices=["openai", "e5"], default="e5",
                    help="Proveedor de embeddings: openai (API) o e5 (local)")
    ap.add_argument("--openai-model", default="text-embedding-3-small",
                    help="Modelo OpenAI (default: text-embedding-3-small)")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base",
                    help="Modelo E5 local (default: multilingual-e5-base)")
    ap.add_argument("--batch-size", type=int, default=256, help="Tamaño de lote para indexar (default 256)")
    ap.add_argument("--skip-existing", action="store_true", help="Saltar IDs ya existentes en la colección")
    ap.add_argument("--compose", choices=["with_meta", "plain"], default="with_meta",
                    help="Cómo componer el texto a embeder (default with_meta)")
    args = ap.parse_args()

    jsonl_path = Path(args.file).expanduser().resolve()
    if not jsonl_path.exists():
        print(f"[ERR] No existe el archivo: {jsonl_path}")
        return 2

    persist_dir = Path(args.persist_dir).expanduser().resolve()

    index_jsonl_to_chroma(
        jsonl_path=jsonl_path,
        persist_dir=persist_dir,
        collection=args.collection,
        provider=args.provider,
        openai_model=args.openai_model,
        e5_model=args.e5_model,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        compose_mode=args.compose,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
