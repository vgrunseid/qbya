#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_auto.py — Orquestador RAG para tu pipeline (etapa 4):
- Conecta a Chroma ya indexado (paso 3).
- Recupera con auto-detección de droga/sección + multi-queries + RRF.
- Rerank (keyword/cross), vecinos ±N y respuesta con LLM (Ollama/OpenAI) con citas.

Uso rápido desde Python:
    from rag_auto import RAG
    rag = RAG(persist_dir="chroma/prospectos", collection="prospectos",
              provider="e5", e5_model="intfloat/multilingual-e5-base",
              llm_backend="ollama", llm_model="llama3.1")
    ans, docs = rag.answer("posología de alplax xr", return_docs=True)
    print(ans)

Requisitos: langchain-chroma, sentence-transformers (si provider=e5), FlagEmbedding (si rerank=cross)
"""

from __future__ import annotations
import os, re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------- Sinónimos de secciones -----------------
SECTION_SYNONYMS: Dict[str, List[str]] = {
    "propiedades farmacológicas": [
        "PROPIEDADES FARMACOLÓGICAS","PROPIEDADES FARMACODINÁMICAS","MECANISMO DE ACCIÓN","FARMACODINAMIA",
    ],
    "posología": [
        "POSOLOGÍA Y MODO DE ADMINISTRACIÓN","POSOLOGÍA","DOSIS","MODO DE ADMINISTRACIÓN",
    ],
    "contraindicaciones": ["CONTRAINDICACIONES","CONTRA-INDICACIONES"],
    "advertencias y precauciones": ["ADVERTENCIAS Y PRECAUCIONES","ADVERTENCIAS","PRECAUCIONES"],
    "reacciones adversas": ["REACCIONES ADVERSAS","EVENTOS ADVERSOS","EFECTOS ADVERSOS"],
    "interacciones": ["INTERACCIONES","INTERACCIÓN CON OTRAS DROGAS","INTERACCIÓN CON OTROS MEDICAMENTOS"],
}

def _syns(hint: Optional[str]) -> List[str]:
    return SECTION_SYNONYMS.get((hint or "").lower().strip(), [hint] if hint else [])

# ----------------- Embeddings -----------------
def _make_embeddings(provider: str, openai_model: str, e5_model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Falta OPENAI_API_KEY para provider=openai.")
        return OpenAIEmbeddings(model=openai_model)
    if provider == "e5":
        class E5Emb:
            def __init__(self, name: str):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(name, device="cpu")
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.model.encode([f"passage: {t}" for t in texts],
                                         normalize_embeddings=True, convert_to_numpy=True).tolist()
            def embed_query(self, text: str) -> List[float]:
                return self.model.encode([f"query: {text}"],
                                         normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        return E5Emb(e5_model)
    raise ValueError("provider debe ser 'openai' o 'e5'")

# ----------------- LLM -----------------
def _make_llm(backend: str, model: str, temperature: float = 0.1):
    if backend == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model, temperature=temperature)
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError("backend LLM no soportado")

# ----------------- Recuperación helpers -----------------
def _unique_drugs(vs: Chroma) -> List[str]:
    try:
        metas = vs._collection.get(include=["metadatas"], limit=1_000_000).get("metadatas", [])
        return sorted({(m or {}).get("drug_name", "") for m in metas if (m or {}).get("drug_name")})
    except Exception:
        return []

def _auto_drug_from_query(vs: Chroma, query: str) -> Optional[str]:
    q = query.lower()
    candidates = [d for d in _unique_drugs(vs) if d and d in q]
    return candidates[0] if candidates else None

def _auto_section_hint(query: str) -> Optional[str]:
    q = query.lower()
    for k, syns in SECTION_SYNONYMS.items():
        if any(s.lower() in q for s in [k] + syns):
            return k
    return None

def _build_md_filter(drug: Optional[str]) -> Optional[Dict[str, Any]]:
    return {"drug_name": {"$eq": drug.lower()}} if drug else None

def _build_where_doc(section_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    syns = _syns(section_hint)
    return {"$or": [{"$contains": s} for s in syns]} if syns else None

def _expand_queries(q: str, section_hint: Optional[str]) -> List[str]:
    out = [q]
    syns = _syns(section_hint)
    if syns:
        out += [f"{s} {q}" for s in syns][:5]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:6]

def _similarity_search(vs: Chroma, query: str, k: int,
                       md_filter: Optional[Dict[str, Any]],
                       where_doc: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
    kwargs: Dict[str, Any] = {"k": max(k, 10)}
    if md_filter: kwargs["filter"] = md_filter
    if where_doc: kwargs["where_document"] = where_doc
    try:
        return vs.similarity_search_with_score(query, **kwargs)
    except Exception as e:
        if "where_document" in str(e):
            kwargs.pop("where_document", None)
            return vs.similarity_search_with_score(query, **kwargs)
        print(f"⚠️ Búsqueda falló: {type(e).__name__}: {e}")
        return []

def _rrf_merge(list_of_lists: List[List[Tuple[Document, float]]], k: int) -> List[Tuple[Document, float]]:
    key_for = lambda d: (d.metadata.get("doc_name"),
                         d.metadata.get("section_canonical"),
                         (d.page_content or "")[:120])
    scores = defaultdict(float); first: Dict[Any, Document] = {}
    for hits in list_of_lists:
        for rank, (doc, _raw) in enumerate(hits, 1):
            key = key_for(doc)
            scores[key] += 1.0 / (60.0 + rank)
            if key not in first: first[key] = doc
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(first[key], val) for key, val in ranked]

def _add_neighbors(vs: Chroma, docs_scores: List[Tuple[Document, float]], n: int = 1) -> List[Tuple[Document, float]]:
    if n <= 0 or not docs_scores: return docs_scores
    out = list(docs_scores)
    seen = set((d.metadata.get("doc_name"), (d.page_content or "")[:120]) for d,_ in out)
    for doc,_ in docs_scores[:2]:
        md = doc.metadata or {}; doc_name = md.get("doc_name")
        if not doc_name: continue
        try:
            res = vs._collection.get(where={"doc_name": {"$eq": doc_name}}, include=["documents","metadatas"])
        except Exception:
            continue
        docs = res.get("documents", []) or []; metas = res.get("metadatas", []) or []
        try:
            idx = docs.index(doc.page_content or "")
        except ValueError:
            continue
        for d in range(1, n+1):
            for j in (idx-d, idx+d):
                if 0 <= j < len(docs):
                    key = (metas[j].get("doc_name"), (docs[j] or "")[:120])
                    if key in seen: continue
                    out.append((Document(page_content=docs[j], metadata=metas[j]), 0.0))
                    seen.add(key)
    return out

def _keyword_score(q: str, text: str, bonus_terms: Iterable[str] = ()) -> float:
    txt = text.lower()
    toks = re.findall(r"\w+", q.lower())
    bonus = [t.lower() for t in bonus_terms]
    score = sum(txt.count(t) for t in toks) + 2.0 * sum(txt.count(b) for b in bonus)
    return float(score)

def _apply_rerank(docs_scores: List[Tuple[Document, float]],
                  mode: str, query: str, section_hint: Optional[str],
                  rerank_model: str) -> List[Tuple[Document, float]]:
    if not docs_scores or mode == "none": return docs_scores
    if mode == "keyword":
        syns = _syns(section_hint)
        rescored = [(doc, _keyword_score(query, doc.page_content or "", syns)) for doc,_ in docs_scores]
        return sorted(rescored, key=lambda x: x[1], reverse=True)
    if mode == "cross":
        try:
            from FlagEmbedding import FlagReranker
            rer = FlagReranker(rerank_model, use_fp16=True)
            scores = rer.compute_score([(query, d.page_content or "") for d,_ in docs_scores])
            rescored = list(zip([d for d,_ in docs_scores], scores))
            return sorted(rescored, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"⚠️ Cross-rerank no disponible ({e}). Caigo a 'keyword'.")
            return _apply_rerank(docs_scores, "keyword", query, section_hint, rerank_model)
    return docs_scores

def _infer_majority_drug(docs: List[Document]) -> Optional[str]:
    drugs = [(d.metadata or {}).get("drug_name") for d in docs if (d.metadata or {}).get("drug_name")]
    if not drugs: return None
    top, cnt = Counter(drugs).most_common(1)[0]
    return top if (cnt / max(1, len(drugs))) >= 0.6 else None

def _build_answer(llm, query: str, section_hint: Optional[str], docs: List[Document]) -> str:
    ctx_parts = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
        head = f"[{i}] {md.get('drug_name')} • {sec} • {md.get('doc_name')}"
        ctx_parts.append(head + "\n" + (d.page_content or ""))
    context = "\n\n-----\n\n".join(ctx_parts[:6])
    focus = f" Enfocate en la sección: {section_hint.upper()}." if section_hint else ""
    prompt = f"""Sos un asistente para información de prospectos médicos. Respondé de forma breve, factual y con citas [#].{focus}
Si la pregunta no está respondida por el contexto, decí explícitamente que el contexto no alcanza.

Pregunta: {query}

Contexto:
{context}

Instrucciones:
- No inventes información ni asumas datos que no estén en el contexto.
- Resumí en oraciones claras. Si hay listas, integrá los puntos clave.
- Mostrá 1–4 referencias al final como [1], [2], … usando los índices de los fragmentos provistos.
- Si la pregunta no menciona un medicamento y el contexto no muestra claramente uno único, indicá que no hay información suficiente.
Respuesta:"""
    return llm.invoke(prompt)

# ----------------- Clase RAG -----------------
class RAG:
    def __init__(self,
                 persist_dir: str = "chroma/prospectos",
                 collection: str = "prospectos",
                 provider: str = "e5",
                 openai_model: str = "text-embedding-3-small",
                 e5_model: str = "intfloat/multilingual-e5-base",
                 llm_backend: str = "ollama",
                 llm_model: str = "llama3.1"):
        self.emb = _make_embeddings(provider, openai_model, e5_model)
        self.vs = Chroma(collection_name=collection,
                         persist_directory=persist_dir,
                         embedding_function=self.emb)
        self.llm = _make_llm(llm_backend, llm_model)

    # — Recuperación de alto nivel —
    def retrieve(self, query: str, topk: int = 6, neighbors: int = 1,
                 rerank: str = "keyword", rerank_model: str = "BAAI/bge-reranker-v2-m3") -> List[Tuple[Document, float]]:
        query = (query or "").strip()
        # 1) auto detecciones
        drug_auto = _auto_drug_from_query(self.vs, query)
        sec_hint = _auto_section_hint(query)
        md_filter = _build_md_filter(drug_auto)
        where_doc = _build_where_doc(sec_hint)
        queries = _expand_queries(query, sec_hint)

        # 2) multi-queries + RRF
        all_hits: List[List[Tuple[Document, float]]] = []
        for qi in queries:
            hits = _similarity_search(self.vs, qi, topk, md_filter, where_doc)
            all_hits.append(hits)
        docs_scores = _rrf_merge(all_hits, topk)

        # 3) si no hay droga en query, inferir mayoría y refinar
        if not drug_auto and docs_scores:
            maj = _infer_majority_drug([d for d,_ in docs_scores])
            if maj:
                md_filter = _build_md_filter(maj)
                all_hits = []
                for qi in queries:
                    hits = _similarity_search(self.vs, qi, topk, md_filter, where_doc)
                    all_hits.append(hits)
                docs_scores = _rrf_merge(all_hits, topk)

        # 4) vecinos + rerank
        if neighbors > 0:
            docs_scores = _add_neighbors(self.vs, docs_scores, n=neighbors)
        docs_scores = _apply_rerank(docs_scores, rerank, query, sec_hint, rerank_model)
        return docs_scores

    def answer(self, query: str, topk: int = 6, neighbors: int = 1,
               rerank: str = "keyword", rerank_model: str = "BAAI/bge-reranker-v2-m3",
               return_docs: bool = False):
        docs_scores = self.retrieve(query, topk=topk, neighbors=neighbors,
                                    rerank=rerank, rerank_model=rerank_model)
        if not docs_scores:
            msg = "No encontré fragmentos relevantes en la base. Probá reformular la pregunta."
            return (msg, []) if return_docs else msg
        docs = [d for d,_ in docs_scores[:6]]
        sec_hint = _auto_section_hint(query)
        ans = _build_answer(self.llm, query, sec_hint, docs)
        return (ans, docs) if return_docs else ans
