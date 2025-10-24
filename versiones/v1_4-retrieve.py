#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-retrieve.py — Recuperación "auto" desde Chroma con detección flexible de droga/familia.
- Detecta droga(s) por match exacto, prefijo y fuzzy; expande a la "familia" (marca y variantes).
- Multi-queries con sinónimos de sección + fusión RRF.
- Re-rank ligero por keywords (default) o cross-encoder opcional.
- Vecinos ±1 del top para dar contexto.
- SIEMPRE agrega una búsqueda base por proximidad (sin filtros) para no quedarse sin resultados.
- Respuesta con LLM (Ollama por defecto) con citas [#].

NUEVO:
  --output {full,answer}
    - full   (default): imprime todo como siempre.
    - answer: SOLO imprime la respuesta del LLM (stdout limpio). Si no hay, imprime el mensaje
              "No hay información suficiente en el contexto." y sale con código 2.

Ejemplos:
  python 4-retrieve.py -q "posología de alplax xr"
  python 4-retrieve.py -q "propiedades farmacológicas de ozempic" --output full --answer
  python 4-retrieve.py -q "contraindicaciones de gadopril" --output answer --answer
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import difflib
import unicodedata
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------- Sinónimos de secciones -----------------
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
    "reacciones adversas": [
        "REACCIONES ADVERSAS",
        "EVENTOS ADVERSOS",
        "EFECTOS ADVERSOS",
    ],
    "interacciones": [
        "INTERACCIONES",
        "INTERACCIÓN CON OTRAS DROGAS",
        "INTERACCIÓN CON OTROS MEDICAMENTOS",
    ],
}

# ----------------- Helpers de logging -----------------
def make_emitter(output_mode: str):
    """Devuelve una función print-like que NO imprime si output=answer."""
    quiet = (output_mode == "answer")
    def emit(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)
    return emit

# ----------------- Normalización / familia -----------------
def _norm(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def brand_root(name: str) -> str:
    """
    Raíz “de marca” simple: tramo alfabético inicial hasta separadores comunes (espacio, guion, dígitos).
    Ej: 'alplax digest' -> 'alplax'
    """
    n = _norm(name)
    m = re.match(r"[a-z]+", n)
    return m.group(0) if m else n

# ----------------- Embeddings -----------------
def make_embeddings(provider: str, openai_model: str, e5_model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Definí OPENAI_API_KEY para usar provider=openai.")
        return OpenAIEmbeddings(model=openai_model)
    if provider == "e5":
        class E5Emb:
            def __init__(self, name: str):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(name, device="cpu")
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                texts = [f"passage: {t}" for t in texts]
                return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
            def embed_query(self, text: str) -> List[float]:
                q = f"query: {text}"
                return self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        return E5Emb(e5_model)
    raise ValueError("provider debe ser 'openai' o 'e5'.")

# ----------------- Index helpers -----------------
def unique_drugs(vs: Chroma) -> List[str]:
    try:
        metas = vs._collection.get(include=["metadatas"], limit=1_000_000).get("metadatas", [])
        return sorted({(m or {}).get("drug_name","") for m in metas if (m or {}).get("drug_name")})
    except Exception:
        return []

def unique_drugs_norm(vs: Chroma) -> List[Tuple[str, str, str]]:
    """
    Lista de drogas de índice: (drug_name_raw, norm, root)
    """
    try:
        metas = vs._collection.get(include=["metadatas"], limit=1_000_000).get("metadatas", [])
    except Exception:
        metas = []
    seen = {}
    for m in metas:
        dn = (m or {}).get("drug_name") or ""
        if not dn:
            continue
        if dn in seen:
            continue
        seen[dn] = True
    out: List[Tuple[str,str,str]] = []
    for raw in seen.keys():
        n = _norm(raw)
        r = brand_root(raw)
        out.append((raw, n, r))
    return out

# ----------------- Detección flexible de droga -----------------
def detect_drug_candidates(vs: Chroma, query: str) -> List[str]:
    """
    Devuelve lista de candidatos (drug_name tal como está en metadatos) a partir de:
    - match por palabra exacta (boundaries),
    - prefijo (>=4),
    - fuzzy con difflib (cutoff 0.87),
    y expande a la FAMILIA (misma brand_root).
    """
    qn = _norm(query)
    drugs = unique_drugs_norm(vs)  # [(raw, norm, root)]
    if not drugs:
        return []

    # 1) exact word match
    exact = []
    for raw, n, r in drugs:
        if re.search(rf"\b{re.escape(n)}\b", qn):
            exact.append((raw, n, r))

    # 2) prefix (>= 4 chars)
    pref = []
    if not exact:
        toks = [t for t in re.findall(r"[a-z0-9]+", qn) if len(t) >= 4]
        if toks:
            tset = set(toks)
            for raw, n, r in drugs:
                if any(n.startswith(t) for t in tset):
                    pref.append((raw, n, r))

    # 3) fuzzy
    fuzz = []
    if not exact and not pref:
        toks = [t for t in re.findall(r"[a-z0-9]+", qn) if len(t) >= 4]
        for t in toks:
            candidates = difflib.get_close_matches(t, [n for _, n, _ in drugs], n=5, cutoff=0.87)
            if candidates:
                cand_set = set(candidates)
                fuzz += [(raw, n, r) for raw, n, r in drugs if n in cand_set]

    base = exact or pref or fuzz
    if not base:
        return []

    # 4) expand to family by root
    roots = {r for _, _, r in base}
    family = [raw for raw, n, r in drugs if r in roots]

    # dedup ordenando por longitud desc para priorizar variantes más específicas
    family = list(dict.fromkeys(sorted(family, key=lambda s: -len(s))))
    return family[:50]

# ----------------- Secciones -----------------
def auto_detect_section_hint(query: str) -> Optional[str]:
    q = _norm(query)
    for k, syns in SECTION_SYNONYMS.items():
        if any(_norm(s) in q for s in ([k] + syns)):
            return k
    return None

def synonyms_for(hint: Optional[str]) -> List[str]:
    if not hint:
        return []
    return SECTION_SYNONYMS.get(hint.lower().strip(), [hint])

# ----------------- Filtros -----------------
def build_md_filter_multi(drug_list: List[str]) -> Optional[Dict[str, Any]]:
    if not drug_list:
        return None
    # IMPORTANTE: no forzar lowercase; usar tal cual está en metadatos.
    return {"$or": [{"drug_name": {"$eq": d}} for d in drug_list]}

def build_where_document(section_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    syns = synonyms_for(section_hint)
    if not syns:
        return None
    return {"$or": [{"$contains": s} for s in syns]}

def build_where_document_drug(drug_list: List[str]) -> Optional[Dict[str, Any]]:
    if not drug_list:
        return None
    return {"$or": [{"$contains": d} for d in drug_list]}

def post_filter_by_drug_family(docs_scores: List[Tuple[Document, float]], drug_list: List[str]) -> List[Tuple[Document, float]]:
    if not docs_scores or not drug_list:
        return docs_scores
    fam_roots = {_norm(brand_root(d)) for d in drug_list}
    out: List[Tuple[Document, float]] = []
    for doc, score in docs_scores:
        dn = _norm((doc.metadata or {}).get("drug_name") or "")
        if _norm(brand_root(dn)) in fam_roots:
            out.append((doc, score))
    return out or docs_scores

# ----------------- Recuperación -----------------
def expand_queries(q: str, section_hint: Optional[str]) -> List[str]:
    out = [q]
    syns = synonyms_for(section_hint)
    if syns:
        out += [f"{s} {q}" for s in syns][:5]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:6]

def similarity_search_safe(vs: Chroma, query: str, k: int,
                           md_filter: Optional[Dict[str, Any]],
                           where_doc: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
    kwargs: Dict[str, Any] = {"k": max(k, 10)}
    if md_filter: kwargs["filter"] = md_filter
    if where_doc: kwargs["where_document"] = where_doc
    try:
        hits = vs.similarity_search_with_score(query, **kwargs)
        # fallback: si no devuelve nada y había filtros, reintentar sin filtros
        if not hits and ("filter" in kwargs or "where_document" in kwargs):
            kwargs.pop("filter", None)
            kwargs.pop("where_document", None)
            hits = vs.similarity_search_with_score(query, **kwargs)
        return hits
    except Exception:
        # reintentar sin where_document
        kwargs.pop("where_document", None)
        try:
            hits = vs.similarity_search_with_score(query, **kwargs)
            if not hits and "filter" in kwargs:
                kwargs.pop("filter", None)
                hits = vs.similarity_search_with_score(query, **kwargs)
            return hits
        except Exception:
            return []

def rrf_merge(list_of_lists: List[List[Tuple[Document, float]]], k: int) -> List[Tuple[Document, float]]:
    key_for = lambda d: (d.metadata.get("doc_name"), d.metadata.get("section_canonical"), (d.page_content or "")[:120])
    scores = defaultdict(float); first: Dict[Any, Document] = {}
    for hits in list_of_lists:
        for rank, (doc, _raw) in enumerate(hits, 1):
            key = key_for(doc)
            scores[key] += 1.0 / (60.0 + rank)
            if key not in first: first[key] = doc
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(first[key], val) for key, val in ranked]

def add_neighbors(vs: Chroma, docs_scores: List[Tuple[Document, float]], n: int = 1) -> List[Tuple[Document, float]]:
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

def keyword_score(q: str, text: str, bonus_terms: Iterable[str] = ()) -> float:
    txt = text.lower()
    toks = re.findall(r"\w+", q.lower())
    bonus = [t.lower() for t in bonus_terms]
    score = sum(txt.count(t) for t in toks)
    score += 2.0 * sum(txt.count(b) for b in bonus)
    return float(score)

def apply_rerank(docs_scores: List[Tuple[Document, float]],
                 mode: str,
                 query: str,
                 section_hint: Optional[str],
                 rerank_model: str) -> List[Tuple[Document, float]]:
    if not docs_scores or mode == "none":
        return docs_scores
    if mode == "keyword":
        syns = synonyms_for(section_hint)
        rescored = [(doc, keyword_score(query, doc.page_content or "", syns)) for doc,_ in docs_scores]
        return sorted(rescored, key=lambda x: x[1], reverse=True)
    if mode == "cross":
        try:
            from FlagEmbedding import FlagReranker
            rer = FlagReranker(rerank_model, use_fp16=True)
            pairs = [(query, d.page_content or "") for d,_ in docs_scores]
            scores = rer.compute_score(pairs)
            rescored = list(zip([d for d,_ in docs_scores], scores))
            return sorted(rescored, key=lambda x: x[1], reverse=True)
        except Exception:
            return apply_rerank(docs_scores, "keyword", query, section_hint, rerank_model)
    return docs_scores

def print_results(docs_scores: List[Tuple[Document, float]], emit) -> None:
    if not docs_scores:
        emit("(sin resultados)")
        return
    for i, (doc, score) in enumerate(docs_scores, 1):
        md = doc.metadata or {}
        sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
        snippet = (doc.page_content or "").replace("\n", " ")
        emit(f"\n[{i}] score={score:.4f} | {md.get('drug_name')} | {sec} | {md.get('doc_name')}")
        emit(f"     {snippet[:220]}{'…' if len(snippet)>220 else ''}")

# ----------------- LLM -----------------
def make_llm(backend: str, model: str, temperature: float = 0.1):
    if backend == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model, temperature=temperature)
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError("backend LLM no soportado")

def infer_majority_drug(docs: List[Document]) -> Optional[str]:
    drugs = [ (d.metadata or {}).get("drug_name") for d in docs if (d.metadata or {}).get("drug_name") ]
    if not drugs: return None
    c = Counter(drugs)
    top, cnt = c.most_common(1)[0]
    return top if (cnt / max(1, len(drugs))) >= 0.6 else None

def build_answer(llm, query: str, section_hint: Optional[str], docs: List[Document]) -> str:
    # contexto compacto
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
- No inventes información ni asumas datos que no estén en el contexto. No uses conocimiento externo.
- Si una frase no puede respaldarse con fragmentos del contexto, omitila.
- Mencioná explícitamente el nombre exacto del medicamento según el metadato `drug_name` de los fragmentos citados.
- Si la consulta incluye varios medicamentos, organizá la respuesta en subsecciones por medicamento, cada una con su `drug_name`.
- Resumí en oraciones claras y lenguaje para consumidor final; explicá términos técnicos entre paréntesis si aparecen en el contexto.
- Responde solo a lo que se pregunta; incluye únicamente información necesaria y directamente relacionada (puedes añadir detalles que aclaren la respuesta). No agregues información no solicitada. Si el dato no está en el CONTEXTO, escribe: “No hay información suficiente para responder con precisión.”
- No des consejos médicos personalizados; limitate a lo que figura en los prospectos. Podés cerrar con una advertencia general (p. ej., consultar a un profesional ante dudas).
- No extrapoles a otras presentaciones/dosis/indicaciones si no figuran en los fragmentos citados.
- Mostrá 1–4 referencias al final como [1], [2], … usando los índices de los fragmentos provistos (en el orden que aparecen en el bloque Contexto).
- Si la pregunta no menciona un medicamento y el contexto no muestra claramente un único fármaco, indicá que no hay información suficiente para responder con precisión.
- Si hay inconsistencias entre fragmentos, indicá que el contexto es insuficiente o contradictorio para responder con precisión.

Respuesta:"""
    return llm.invoke(prompt)

# ----------------- MAIN -----------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Q&A simplificado sobre prospectos (solo pregunta).")
    ap.add_argument("-q", "--query", required=True, help="Pregunta del usuario en lenguaje natural")

    # Chroma
    ap.add_argument("-p", "--persist-dir", default="chroma/prospectos", help="Directorio de Chroma (default: chroma/prospectos)")
    ap.add_argument("-c", "--collection", default="prospectos", help="Colección (default: prospectos)")

    # Embeddings (coincidir con los usados al indexar)
    ap.add_argument("--provider", choices=["openai","e5"], default="e5")
    ap.add_argument("--openai-model", default="text-embedding-3-small")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base")

    # Recuperación
    ap.add_argument("-k", "--topk", type=int, default=6)
    ap.add_argument("--neighbors", type=int, default=1)
    ap.add_argument("--rerank", choices=["none","keyword","cross"], default="keyword")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")

    # Respuesta
    ap.add_argument("--answer", action="store_true", help="Si se pasa, genera respuesta con LLM")
    ap.add_argument("--llm-backend", choices=["ollama","openai"], default="ollama")
    ap.add_argument("--llm-model", default="llama3.1")
    ap.add_argument("--debug", action="store_true")

    # Salida
    ap.add_argument("--output", choices=["full","answer"], default="full",
                    help="full: imprime todo; answer: imprime SOLO la respuesta del LLM.")
    args = ap.parse_args()

    emit = make_emitter(args.output)

    # Si pide SOLO respuesta, forzamos generación de respuesta
    if args.output == "answer":
        args.answer = True

    # Conectar a Chroma
    emb = make_embeddings(args.provider, args.openai_model, args.e5_model)
    vs = Chroma(collection_name=args.collection,
                persist_directory=args.persist_dir,
                embedding_function=emb)

    # Sanity
    try:
        count = vs._collection.count()
    except Exception:
        count = len(vs._collection.get(limit=1_000_000).get("ids", []))
    emit(f"→ Colección '{args.collection}' en '{args.persist_dir}' con {count} vectores")
    if count == 0:
        if args.output == "answer":
            print("No hay información suficiente en el contexto.")
            return 2
        emit("⚠️ Colección vacía. Reindexá con embed.py")
        return 1

    # ---- Detección flexible de droga y sección ----
    query = (args.query or "").strip()
    drug_candidates = detect_drug_candidates(vs, query)           # lista de marcas/variantes/familias
    sec_hint = auto_detect_section_hint(query)

    if args.debug:
        emit(f"→ Drogas detectadas (familia): {drug_candidates or 'ninguna'} | section_hint={sec_hint}")

    # Filtros (metadata + refuerzo por contenido)
    md_filter = build_md_filter_multi(drug_candidates)
    where_sec = build_where_document(sec_hint)
    where_drug = build_where_document_drug(drug_candidates)
    where_and = {"$and": [wd for wd in (where_sec, where_drug) if wd]} if (where_sec or where_drug) else None

    # -------------------------------------------------
    # 1ª pasada: SIEMPRE agregar una búsqueda base sin filtros (plan C)
    # -------------------------------------------------
    queries = expand_queries(query, sec_hint)
    all_hits: List[List[Tuple[Document, float]]] = []

    try:
        base_hits = vs.similarity_search_with_score(query, k=max(args.topk, 10))
    except Exception:
        base_hits = []
    all_hits.append(base_hits)

    # Búsquedas con filtros (si hay)
    for qi in queries:
        hits = similarity_search_safe(
            vs, qi, args.topk,
            md_filter=md_filter,
            where_doc=where_and
        )
        all_hits.append(hits)

    docs_scores = rrf_merge(all_hits, args.topk)

    # Post-filter por familia si hay candidatos
    docs_scores = post_filter_by_drug_family(docs_scores, drug_candidates)

    # 2ª pasada opcional: si no hubo candidatos pero hay mayoría clara en resultados, refinar
    if not drug_candidates and docs_scores:
        maj = infer_majority_drug([d for d,_ in docs_scores])
        if args.debug:
            emit(f"→ Mayoría por resultados: {maj or 'ninguna clara'}")
        if maj:
            md_filter2 = build_md_filter_multi([maj])
            where_drug2 = build_where_document_drug([maj])
            where_and2 = {"$and": [wd for wd in (where_sec, where_drug2) if wd]} if (where_sec or where_drug2) else None
            all_hits = []
            # incluir también búsqueda base de nuevo
            try:
                base_hits2 = vs.similarity_search_with_score(query, k=max(args.topk, 10))
            except Exception:
                base_hits2 = []
            all_hits.append(base_hits2)
            for qi in queries:
                hits = similarity_search_safe(
                    vs, qi, args.topk,
                    md_filter=md_filter2,
                    where_doc=where_and2
                )
                all_hits.append(hits)
            docs_scores = rrf_merge(all_hits, args.topk)

    # Vecinos y re-rank
    if args.neighbors > 0:
        docs_scores = add_neighbors(vs, docs_scores, n=args.neighbors)
    docs_scores = apply_rerank(docs_scores, args.rerank, query, sec_hint, args.rerank_model)

    # Si después de todo no hay contexto, comportarse bien según modo de salida
    if not docs_scores:
        msg = "No hay información suficiente en el contexto."
        if args.output == "answer":
            print(msg)
            return 2
        else:
            emit("(sin resultados)")
            emit(msg)
            return 0

    # Mostrar resultados (solo en modo "full")
    if args.output == "full":
        print_results(docs_scores, emit)

    # Responder con LLM
    if args.answer and docs_scores:
        try:
            llm = make_llm(args.llm_backend, args.llm_model)
        except TypeError:
            llm = make_llm(getattr(args, "llm_backend"), getattr(args, "llm_model"))
        try:
            ans = build_answer(llm, query, sec_hint, [d for d,_ in docs_scores[:6]])
            if args.output == "answer":
                print(str(ans).strip())
            else:
                emit("\n— Generando respuesta con LLM —\n")
                emit("Respuesta:\n" + str(ans).strip())
        except Exception as e:
            if args.output == "answer":
                print("No hay información suficiente en el contexto.")
                return 2
            emit(f"⚠️ No pude generar respuesta: {type(e).__name__}: {e}")
    elif args.output == "answer":
        # pidió solo respuesta pero no hay docs
        print("No hay información suficiente en el contexto.")
        return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
