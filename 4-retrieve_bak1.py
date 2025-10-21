#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-retrieve.py — Recuperación desde Chroma con hints de sección, multi-queries, RRF y re-rank.

Ejemplos:
  # E5 local + hint de sección + re-rank por cross-encoder + vecinos
  python 4-retrieve.py -q "propiedades farmacológicas de ozempic" \
    -p chroma/prospectos -c prospectos \
    --provider e5 --e5-model intfloat/multilingual-e5-base \
    --drug ozempic --section-hint "propiedades farmacológicas" \
    --expand-queries --rerank cross --rerank-model BAAI/bge-reranker-v2-m3 \
    --neighbors 1 -k 8 --answer --llm-backend ollama --llm-model llama3.1

  # OpenAI embeddings + keyword rerank
  python 4-retrieve.py -q "posología de sertal" \
    -p chroma/prospectos -c prospectos \
    --provider openai --openai-model text-embedding-3-small \
    --drug "sertal gotas" --rerank keyword -k 6
"""
from __future__ import annotations
import argparse
import os
import sys
import re
from typing import List, Dict, Any, Tuple, Iterable
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------- embeddings ----------
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

    raise ValueError("provider debe ser 'openai' o 'e5'")

# ---------- sinónimos de secciones (podés extenderlos) ----------
SECTION_SYNONYMS: Dict[str, List[str]] = {
    "propiedades farmacológicas": [
        "PROPIEDADES FARMACOLÓGICAS",
        "PROPIEDADES FARMACODINÁMICAS",
        "MECANISMO DE ACCIÓN",
        "FARMACODINAMIA",
    ],
    "posología": [
        "POSOLOGÍA Y MODO DE ADMINISTRACIÓN",
        "POSOLOGÍA",
        "DOSIS",
        "MODO DE ADMINISTRACIÓN",
    ],
    "contraindicaciones": [
        "CONTRAINDICACIONES",
        "CONTRA-INDICACIONES",
    ],
    "advertencias y precauciones": [
        "ADVERTENCIAS Y PRECAUCIONES",
        "ADVERTENCIAS",
        "PRECAUCIONES",
    ],
}

def normalize(s: str | None) -> str:
    return (s or "").strip()

def build_metadata_filter(drug: str | None, section: str | None) -> Dict[str, Any] | None:
    """Filtro por metadatos. Usar operadores válidos ($eq/$in...)."""
    clauses: List[Dict[str, Any]] = []
    if drug:
        clauses.append({"drug_name": {"$eq": drug.lower()}})
    if section:
        sec_up = section.upper()
        clauses.append({"$or": [
            {"section_canonical": {"$in": [sec_up]}},
            {"section_raw": {"$in": [sec_up]}},
        ]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def synonyms_for(section_hint: str | None) -> List[str]:
    if not section_hint:
        return []
    key = section_hint.lower().strip()
    return SECTION_SYNONYMS.get(key, [section_hint])

def build_where_document(section_hint: str | None) -> Dict[str, Any] | None:
    """where_document usa $contains en el TEXTO (no metadatos).
    Intentamos OR de sinónimos; si tu Chroma no soporta $or aquí, igual hacemos multi-queries."""
    syns = synonyms_for(section_hint)
    if not syns:
        return None
    return {"$or": [{"$contains": s} for s in syns]}

def expand_queries(q: str, section_hint: str | None) -> List[str]:
    """Genera variantes de la query con sinónimos de la sección."""
    out = [q]
    syns = synonyms_for(section_hint)
    base = q.lower()
    if not syns:
        return out
    # reemplazos suaves: si la query contiene algún sinónimo, generamos pares alternativos
    for s in syns:
        s_low = s.lower()
        if any(tok in base for tok in [s_low, s_low.capitalize(), s_low.title()]):
            # reemplazar por el resto de sinónimos
            for alt in syns:
                if alt == s:
                    continue
                out.append(re.sub(re.escape(s_low), alt, q, flags=re.IGNORECASE))
    # y si no detectamos presencia, igual añadimos variantes "sección + droga"
    if len(out) == 1:
        out.extend([f"{syn} {q}" for syn in syns])
    # dedup conservando orden
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq[:6]  # no exagerar

# ---------- impresión ----------
def print_results(docs_with_scores: List[Tuple[Document, float]]) -> None:
    if not docs_with_scores:
        print("(sin resultados)")
        return
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        md = doc.metadata or {}
        snippet = (doc.page_content or "").replace("\n", " ")
        sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
        print(f"\n[{i}] score={score:.4f} | {md.get('drug_name')} | {sec} | {md.get('doc_name')}")
        print(f"     {snippet[:220]}{'…' if len(snippet)>220 else ''}")

# ---------- fusor RRF ----------
def rrf_merge(list_of_lists: List[List[Tuple[Document, float]]], k: int) -> List[Tuple[Document, float]]:
    """Combina resultados de múltiples consultas por RRF."""
    key_for = lambda d: (d.metadata.get("doc_name"), d.metadata.get("section_canonical"), (d.page_content or "")[:120])
    scores = defaultdict(float)
    first_doc_by_key: Dict[Any, Document] = {}
    for hits in list_of_lists:
        for rank, (doc, _raw) in enumerate(hits, 1):
            key = key_for(doc)
            scores[key] += 1.0 / (60.0 + rank)
            if key not in first_doc_by_key:
                first_doc_by_key[key] = doc
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(first_doc_by_key[key], val) for key, val in ranked]

# ---------- rerankers ----------
def keyword_score(q: str, text: str, extra_terms: Iterable[str] = ()) -> float:
    txt = text.lower()
    toks = re.findall(r"\w+", q.lower())
    bonus = [t.lower() for t in extra_terms]
    score = sum(txt.count(t) for t in toks)
    score += 2.0 * sum(txt.count(b) for b in bonus)
    return float(score)

def apply_rerank(docs_scores: List[Tuple[Document, float]],
                 mode: str,
                 query: str,
                 section_hint: str | None,
                 rerank_model: str) -> List[Tuple[Document, float]]:
    if not docs_scores or mode == "none":
        return docs_scores
    if mode == "keyword":
        syns = synonyms_for(section_hint)
        rescored = [ (doc, keyword_score(query, doc.page_content or "", syns)) for doc,_ in docs_scores ]
        return sorted(rescored, key=lambda x: x[1], reverse=True)
    if mode == "cross":
        try:
            from FlagEmbedding import FlagReranker
        except Exception:
            print("⚠️  No encontré FlagEmbedding. Caigo a 'keyword' rerank.")
            return apply_rerank(docs_scores, "keyword", query, section_hint, rerank_model)
        rer = FlagReranker(rerank_model, use_fp16=True)
        pairs = [(query, d.page_content or "") for d,_ in docs_scores]
        scores = rer.compute_score(pairs)
        rescored = list(zip([d for d,_ in docs_scores], scores))
        return sorted(rescored, key=lambda x: x[1], reverse=True)
    return docs_scores

# ---------- vecinos (±N) dentro del mismo doc_name ----------
def add_neighbors(vs: Chroma,
                  docs_scores: List[Tuple[Document, float]],
                  n: int = 1) -> List[Tuple[Document, float]]:
    if n <= 0 or not docs_scores:
        return docs_scores
    out = list(docs_scores)
    seen_keys = set((d.metadata.get("doc_name"), (d.page_content or "")[:120]) for d,_ in out)
    for doc,_ in docs_scores[:2]:  # expandimos alrededor de los 2 top
        md = doc.metadata or {}
        doc_name = md.get("doc_name")
        if not doc_name:
            continue
        try:
            col = vs._collection
            res = col.get(where={"doc_name": {"$eq": doc_name}},
                          include=["ids","documents","metadatas"])
        except Exception:
            continue
        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        # localizar índice del documento actual por igualdad exacta de texto
        try:
            idx = docs.index(doc.page_content or "")
        except ValueError:
            continue
        for d in range(1, n+1):
            for j in (idx-d, idx+d):
                if 0 <= j < len(docs):
                    key = (metas[j].get("doc_name"), (docs[j] or "")[:120])
                    if key in seen_keys:
                        continue
                    out.append((Document(page_content=docs[j], metadata=metas[j]), 0.0))
                    seen_keys.add(key)
    return out

# ---------- LLMs ----------
def make_llm(backend: str, model: str, temperature: float = 0.1):
    if backend == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model, temperature=temperature)
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError("backend LLM no soportado")

def build_answer(llm, query: str, section_hint: str | None, docs: List[Document]) -> str:
    # armamos un contexto compacto y seguro
    ctx_parts = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
        head = f"[{i}] {md.get('drug_name')} • {sec} • {md.get('doc_name')}"
        ctx_parts.append(head + "\n" + (d.page_content or ""))
    context = "\n\n-----\n\n".join(ctx_parts[:6])

    focus = f" Enfocate en la sección: {section_hint.upper()}." if section_hint else ""
    prompt = f"""Sos un asistente para información de prospectos médicos. Respondé de forma breve, factual y con citas [#].
Si la pregunta no está respondida por el contexto, decí explícitamente que el contexto no alcanza.{focus}

Pregunta: {query}

Contexto:
{context}

Instrucciones:
- No inventes información.
- Si hay listas/tablas, resumí los puntos relevantes en prosa corta.
- Mostrá 1–4 referencias al final como [1], [2], … usando los índices de los fragmentos provistos.
- Si no se especifcan drogas o nombres de medicamento en la pregunta , inficar que no se brindo informacion suficiente para responder a la pregunta y no agregar información a la respuesta

Respuesta:"""
    return llm.invoke(prompt)  # funciona con Ollama y ChatOpenAI

# ---------- main ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Recuperar resultados desde Chroma con hints de sección, multi-queries, RRF y re-rank.")
    ap.add_argument("-q", "--query", required=True, help="Consulta en lenguaje natural")
    ap.add_argument("-p", "--persist-dir", required=True, help="Directorio de persistencia (p.ej. chroma/prospectos)")
    ap.add_argument("-c", "--collection", default="prospectos", help="Nombre de colección")

    # Embeddings
    ap.add_argument("--provider", choices=["openai","e5"], default="e5", help="Embeddings para consultar (mismo que el indexado)")
    ap.add_argument("--openai-model", default="text-embedding-3-small")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base")

    # Top-K y filtros
    ap.add_argument("-k", "--topk", type=int, default=5)
    ap.add_argument("--drug", help="Filtro exacto por drug_name (minúsculas)")
    ap.add_argument("--section", help="Filtro exacto por sección (MAYÚSCULAS)")
    ap.add_argument("--section-hint", help="Hint de sección (ej.: 'propiedades farmacológicas')")
    ap.add_argument("--expand-queries", action="store_true", help="Generar variantes de la consulta usando sinónimos")

    # Rerank y vecinos
    ap.add_argument("--rerank", choices=["none","keyword","cross"], default="none")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--neighbors", type=int, default=0, help="Agregar ±N vecinos del top-1 dentro del mismo documento")

    # Auto filtros (opcional, conserva compatibilidad)
    ap.add_argument("--auto", action="store_true", help="Intentar detectar droga desde la query (simple)")

    # Respuesta con LLM
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--llm-backend", choices=["ollama","openai"], default="ollama")
    ap.add_argument("--llm-model", default="llama3.1")

    args = ap.parse_args()

    emb = make_embeddings(args.provider, args.openai_model, args.e5_model)
    vs = Chroma(collection_name=args.collection,
                persist_directory=args.persist_dir,
                embedding_function=emb)

    # sanity check
    try:
        count = vs._collection.count()
    except Exception:
        count = len(vs._collection.get(limit=1_000_000).get("ids", []))
    print(f"→ Colección '{args.collection}' en '{args.persist_dir}' con {count} vectores")
    if count == 0:
        print("⚠️ La colección está vacía. Reindexá con embed.py.")
        return 1

    # Auto-detector de droga súper simple (si pidieron --auto y no vino --drug)
    drug = normalize(args.drug)
    if args.auto and not drug:
        # sacamos lista de drogas de la colección
        try:
            meta = vs._collection.get(include=["metadatas"], limit=1_000_000).get("metadatas", [])
            uniq = sorted({ (m or {}).get("drug_name","") for m in meta if (m or {}).get("drug_name") })
            qlow = args.query.lower()
            candidates = [d for d in uniq if d in qlow]
            if candidates:
                drug = candidates[0]
        except Exception:
            pass
    if args.auto:
        print(f"→ Auto-Filtros: drug={drug or None} section={args.section or None}")

    # Filtros por metadatos + where_document por hint de sección
    md_filter = build_metadata_filter(drug, args.section)
    if md_filter:
        print(f"→ Filtro Chroma: {md_filter}")
    where_doc = build_where_document(args.section_hint)

    # Lista de queries (multi)
    queries = [args.query]
    if args.expand_queries and args.section_hint:
        queries = expand_queries(args.query, args.section_hint)

    # Búsquedas
    all_hits: List[List[Tuple[Document, float]]] = []
    for qi in queries:
        try:
            kwargs: Dict[str, Any] = {"k": max(args.topk, 10)}
            if md_filter:
                kwargs["filter"] = md_filter
            if where_doc:
                kwargs["where_document"] = where_doc
            hits = vs.similarity_search_with_score(qi, **kwargs)
        except Exception as e:
            # puede fallar por sintaxis de where_document en tu versión de chroma
            if "where_document" in str(e):
                # reintento sin where_document
                kwargs = {"k": max(args.topk, 10)}
                if md_filter:
                    kwargs["filter"] = md_filter
                hits = vs.similarity_search_with_score(qi, **kwargs)
            else:
                print(f"⚠️ Error en búsqueda: {type(e).__name__}: {e}")
                hits = []
        all_hits.append(hits)

    # Fusionamos por RRF y recortamos a top-k
    docs_scores = rrf_merge(all_hits, args.topk)

    # Agregar vecinos
    if args.neighbors > 0:
        docs_scores = add_neighbors(vs, docs_scores, n=args.neighbors)

    # Re-rank
    docs_scores = apply_rerank(docs_scores, args.rerank, args.query, args.section_hint, args.rerank_model)

    # Mostrar
    print_results(docs_scores)

    # Respuesta con LLM
    if args.answer and docs_scores:
        print("\n— Generando respuesta con LLM —\n")
        try:
            llm = make_llm(args.llm_backend, args.llm_model)
            answer = build_answer(llm, args.query, args.section_hint, [d for d,_ in docs_scores[:6]])
            print("Respuesta:\n" + str(answer).strip())
        except Exception as e:
            print(f"⚠️ No pude generar respuesta: {type(e).__name__}: {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
