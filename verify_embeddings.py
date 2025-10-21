#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_embeddings.py — Verifica colección Chroma y recuperación con filtros compatibles.
Usa operadores válidos de Chroma ($eq/$in/...), sin $contains.
"""
from __future__ import annotations
import argparse
import os
from typing import List, Dict, Any, Tuple
from collections import Counter

from langchain_chroma import Chroma  # pip install -U langchain-chroma

# -------- Embeddings providers --------
def make_embeddings(provider: str, openai_model: str, e5_model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Definí OPENAI_API_KEY para provider=openai.")
        return OpenAIEmbeddings(model=openai_model)

    if provider == "e5":
        class E5Emb:
            def __init__(self, model_name: str):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device="cpu")
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                texts = [f"passage: {t}" for t in texts]
                return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
            def embed_query(self, text: str) -> List[float]:
                q = f"query: {text}"
                return self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        return E5Emb(e5_model)

    raise ValueError(f"Provider no soportado: {provider}")

# -------- Helpers de filtros --------
def build_filter(drug: str | None, section: str | None) -> Dict[str, Any] | None:
    """
    Arma un filtro compatible con Chroma 0.5+ usando SOLO operadores válidos.
    - drug -> $eq a minúsculas exacto (así lo guardamos en transform.py)
    - section -> $or de $eq (mayúsculas exacto) y $in (sinónimos opcionales)
    """
    clauses: List[Dict[str, Any]] = []
    if drug:
        clauses.append({"drug_name": {"$eq": drug.lower()}})
    if section:
        sec_up = section.upper()
        # si querés sinónimos exactos, agregalos acá:
        synonyms = [sec_up]
        clauses.append({"$or": [
            {"section_canonical": {"$in": synonyms}},
            {"section_raw": {"$in": synonyms}},
        ]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def list_sections_for_drug(vs: Chroma, drug: str) -> Counter:
    """Consulta directa a Chroma para listar secciones disponibles para una droga."""
    try:
        res = vs._collection.get(
            where={"drug_name": {"$eq": drug.lower()}},
            include=["metadatas"],
            limit=100000
        )
        metas = res.get("metadatas", []) or []
        cnt = Counter()
        for m in metas:
            sec = (m.get("section_canonical") or m.get("section_raw") or "UNKNOWN")
            cnt[sec] += 1
        return cnt
    except Exception:
        return Counter()

# -------- Main --------
def main() -> int:
    ap = argparse.ArgumentParser(description="Verificación de embeddings/Chroma (filtros compatibles).")
    ap.add_argument("-p", "--persist-dir", required=True, help="Directorio de persistencia (p.ej. chroma/prospectos)")
    ap.add_argument("-c", "--collection", default="prospectos", help="Nombre de colección")
    ap.add_argument("--provider", choices=["openai", "e5"], default="openai", help="Embeddings a usar (mismo del indexado).")
    ap.add_argument("--openai-model", default="text-embedding-3-small", help="Modelo OpenAI")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base", help="Modelo E5 local")
    ap.add_argument("-k", "--topk", type=int, default=5, help="Cantidad de resultados por consulta")
    ap.add_argument("--drug", help="Filtro por drug_name (match exacto, minúsculas)")
    ap.add_argument("--section", help="Filtro por sección (match exacto en MAYÚSCULAS)")
    ap.add_argument("--q", "--query", dest="query", help="Consulta única (si no se pasa, usa 3 queries de ejemplo)")
    args = ap.parse_args()

    emb = make_embeddings(args.provider, args.openai_model, args.e5_model)
    vs = Chroma(collection_name=args.collection, persist_directory=args.persist_dir, embedding_function=emb)

    # Conteo
    try:
        count = vs._collection.count()
    except Exception:
        count = len(vs._collection.get(limit=1_000_000).get("ids", []))
    print(f"✔ Colección: {args.collection} | Persist dir: {args.persist_dir} | Vectores: {count}")
    if count == 0:
        print("⚠️ No hay vectores indexados. Revisá 'embed.py'.")
        return 1

    dim = len(emb.embed_query("prueba de dimensión"))
    print(f"✔ Dimensión de embedding: {dim}")

    # Filtro nativo (solo si aplica)
    flt = build_filter(args.drug, args.section)
    if flt:
        print(f"🧩 Filtro Chroma (operadores válidos): {flt}")

    # Queries de prueba
    #queries = [args.query] if args.query else [
    #    "contraindicaciones de ozempic",
    #    "posología de sertal",
    #    "advertencias y precauciones ibupirac",
    #]
    if args.query:
        queries = [args.query]
    elif args.drug and args.section:
        queries = [f"{args.section.lower()} de {args.drug}"]
    elif args.drug:
        queries = [f"información de {args.drug}"]
    else:
        queries = [
            "contraindicaciones de ozempic",
            "posología de sertal",
            "advertencias y precauciones ibupirac",
        ]

    for q in queries:
        print(f"\n🔎 Consulta: {q}")
        # 1) intento con filtro nativo (si existe)
        search_kwargs: Dict[str, Any] = {"k": args.topk}
        if flt:
            search_kwargs["filter"] = flt
        retriever = vs.as_retriever(search_kwargs=search_kwargs)
        try:
            docs = retriever.invoke(q)
        except Exception as e:
            print(f"⚠️ Filtro nativo falló ({type(e).__name__}: {e}). Intento degradar filtros…")
            docs = []

        # 2) si no hubo resultados y tenemos filtro por sección, probá con SOLO droga
        if not docs and args.drug and args.section:
            print("↪︎ Sin resultados con sección. Probando solo por droga…")
            retriever_drug = vs.as_retriever(search_kwargs={
                "k": max(args.topk * 5, 25),
                "filter": {"drug_name": {"$eq": args.drug.lower()}}
            })
            try:
                cand = retriever_drug.invoke(q)
            except Exception:
                cand = retriever.invoke(q)  # sin filtros
            # filtrado manual por sección (contains, case-insensitive) sobre los candidatos
            target = args.section.lower()
            docs = [d for d in cand if target in ((d.metadata or {}).get("section_canonical","").lower()
                                                  or (d.metadata or {}).get("section_raw","").lower())][:args.topk]

        # 3) si aún nada y hay filtro por droga, informá qué secciones existen
        if not docs and args.drug:
            sec_cnt = list_sections_for_drug(vs, args.drug)
            if sec_cnt:
                print(f"ℹ️ Secciones disponibles para '{args.drug.lower()}':")
                for sec, n in sec_cnt.most_common():
                    print(f"   - {sec}: {n}")
            else:
                print(f"ℹ️ No encontré documentos con drug_name='{args.drug.lower()}'.")

        # Mostrar resultados
        if not docs:
            print("  (sin resultados)")
            continue

        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            drug = md.get("drug_name")
            sec = md.get("section_canonical") or md.get("section_raw")
            docn = md.get("doc_name")
            snippet = (d.page_content or "").replace("\n", " ")
            print(f"  [{i}] {drug} | {sec} | {docn}")
            print(f"       {snippet[:180]}{'…' if len(snippet) > 180 else ''}")

    print("\n✅ Verificación terminada.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
