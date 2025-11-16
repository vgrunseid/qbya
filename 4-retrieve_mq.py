#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-retrieve_mq.py

Versión basada en el 4-retrieve original, con:
- detección flexible de droga y sección,
- múltiples consultas (multi-query) a partir de la pregunta base,
- búsqueda por embeddings en Chroma,
- filtrado por metadatos,
- fusión de resultados (RRF),
- vecinos de cada chunk,
- re-ranking (keyword / cross),
- generación de respuesta con LLM (OllamaLLM, como en la versión original),
- impresión en pantalla de PREGUNTA + RESPUESTA (modo full).

Uso típico:
  python 4-retrieve_mq.py -q "¿Contraindicaciones de Alplax?" \
    -p chroma/prospectos -c prospectos \
    --rerank cross --multi-query-rewrites 3 --answer --debug
"""

import argparse
import difflib
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Utilitarios de logging
# ---------------------------------------------------------------------


def emit(msg: str) -> None:
    """Log a stderr (para que stdout quede limpio para la respuesta)."""
    sys.stderr.write(str(msg).rstrip() + "\n")


# ---------------------------------------------------------------------
# Diccionarios y helpers de secciones
# ---------------------------------------------------------------------

SECTION_CANONICAL = {
    "ACCION TERAPEUTICA": "ACCIÓN TERAPÉUTICA",
    "ACCION TERAPEUTICA / GRUPO FARMACOLÓGICO": "ACCIÓN TERAPÉUTICA",
    "ACCIÓN TERAPEUTICA": "ACCIÓN TERAPÉUTICA",
    "ACCIÓN TERAPÉUTICA": "ACCIÓN TERAPÉUTICA",
    "ACCION FARMACOLOGICA": "ACCIÓN TERAPÉUTICA",
    "ACCION FARMACOLÓGICA": "ACCIÓN TERAPÉUTICA",
    "COMPOSICION": "COMPOSICIÓN",
    "COMPOSICIÓN": "COMPOSICIÓN",
    "POSOLOGIA Y MODO DE ADMINISTRACION": "POSOLOGÍA Y MODO DE ADMINISTRACIÓN",
    "POSOLOGÍA Y MODO DE ADMINISTRACIÓN": "POSOLOGÍA Y MODO DE ADMINISTRACIÓN",
    "POSOLOGÍA Y MODO DE ADMINISTRACIÓN / VÍA DE ADMINISTRACIÓN": "POSOLOGÍA Y MODO DE ADMINISTRACIÓN",
    "INDICACIONES": "INDICACIONES",
    "INDICACIONES TERAPEUTICAS": "INDICACIONES",
    "INDICACIONES TERAPÉUTICAS": "INDICACIONES",
    "CONTRAINDICACIONES": "CONTRAINDICACIONES",
    "ADVERTENCIAS Y PRECAUCIONES": "ADVERTENCIAS Y PRECAUCIONES",
    "ADVERTENCIAS": "ADVERTENCIAS Y PRECAUCIONES",
    "PRECAUCIONES": "ADVERTENCIAS Y PRECAUCIONES",
    "REACCIONES ADVERSAS": "REACCIONES ADVERSAS",
    "REACCIONES ADVERSAS / EFECTOS ADVERSOS": "REACCIONES ADVERSAS",
    "INTERACCIONES": "INTERACCIONES",
    "INTERACCIONES MEDICAMENTOSAS": "INTERACCIONES",
    "CONSERVACIÓN / ALMACENAMIENTO": "CONSERVACIÓN / ALMACENAMIENTO",
    "CONSERVACION / ALMACENAMIENTO": "CONSERVACIÓN / ALMACENAMIENTO",
}

SECTION_SYNONYMS = {
    "ACCIÓN TERAPÉUTICA": [
        "acción terapéutica",
        "grupo farmacológico",
        "¿para qué sirve?",
        "para qué se usa",
    ],
    "COMPOSICIÓN": [
        "composición",
        "ingredientes activos",
        "principio activo",
    ],
    "POSOLOGÍA Y MODO DE ADMINISTRACIÓN": [
        "posología",
        "dosis",
        "cómo se toma",
        "cómo se administra",
        "dosis recomendada",
    ],
    "CONTRAINDICACIONES": [
        "contraindicaciones",
        "no debe usarse",
        "en qué casos no usar",
        "cuando está contraindicado",
    ],
    "ADVERTENCIAS Y PRECAUCIONES": [
        "advertencias",
        "precauciones",
        "qué cuidados tener",
        "precauciones especiales",
    ],
    "REACCIONES ADVERSAS": [
        "reacciones adversas",
        "efectos secundarios",
        "efectos adversos",
    ],
    "INTERACCIONES": [
        "interacciones",
        "interacciones medicamentosas",
        "con qué no se puede combinar",
    ],
    "CONSERVACIÓN / ALMACENAMIENTO": [
        "conservación",
        "almacenamiento",
        "cómo conservar",
    ],
}

SECTION_HINT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bcontraindica", re.I), "CONTRAINDICACIONES"),
    (re.compile(r"\bcontraindicaciones\b", re.I), "CONTRAINDICACIONES"),
    (re.compile(r"\bposolog", re.I), "POSOLOGÍA Y MODO DE ADMINISTRACIÓN"),
    (re.compile(r"\bdosis\b", re.I), "POSOLOGÍA Y MODO DE ADMINISTRACIÓN"),
    (re.compile(r"\bcómo se toma\b", re.I), "POSOLOGÍA Y MODO DE ADMINISTRACIÓN"),
    (re.compile(r"\bcomposici[oó]n\b", re.I), "COMPOSICIÓN"),
    (re.compile(r"\bpara qu[eé] se usa\b", re.I), "ACCIÓN TERAPÉUTICA"),
    (re.compile(r"\badvertenc", re.I), "ADVERTENCIAS Y PRECAUCIONES"),
    (re.compile(r"\bprecauc", re.I), "ADVERTENCIAS Y PRECAUCIONES"),
    (re.compile(r"\breacciones adversas\b", re.I), "REACCIONES ADVERSAS"),
    (re.compile(r"\befectos (secundarios|adversos)", re.I), "REACCIONES ADVERSAS"),
    (re.compile(r"\binteracciones\b", re.I), "INTERACCIONES"),
    (re.compile(r"\bconservaci[oó]n\b", re.I), "CONSERVACIÓN / ALMACENAMIENTO"),
    (re.compile(r"\balmacenamiento\b", re.I), "CONSERVACIÓN / ALMACENAMIENTO"),
]


def canonical_section_name(name: str) -> str:
    if not name:
        return "UNKNOWN"
    key = re.sub(r"\s+", " ", name).strip().upper()
    return SECTION_CANONICAL.get(key, key)


def synonyms_for(section: Optional[str]) -> List[str]:
    if not section:
        return []
    can = canonical_section_name(section)
    return SECTION_SYNONYMS.get(can, [])


def auto_detect_section_hint(query: str) -> Optional[str]:
    q = query or ""
    for pat, sec in SECTION_HINT_PATTERNS:
        if pat.search(q):
            return sec
    return None


# ---------------------------------------------------------------------
# Detección de droga y helpers
# ---------------------------------------------------------------------


def normalize_drug_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúñü]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def guess_drug_family(drug: str) -> str:
    d = normalize_drug_name(drug)
    d = re.sub(r"\s+sr\b", "", d)
    d = re.sub(r"\s+xr\b", "", d)
    d = re.sub(r"\s+retard\b", "", d)
    d = re.sub(r"\s+lp\b", "", d)
    d = re.sub(r"\bcomp\.?\b", "", d)
    d = re.sub(r"\bcomprimidos?\b", "", d)
    d = re.sub(r"\scápsulas?\b", "", d)
    d = re.sub(r"\s+\d+(mg|mcg|g|ml)\b", "", d)
    d = re.sub(r"\s+(\d+x\d+)\b", "", d)
    d = re.sub(r"\s+", " ", d).strip()
    return d


def build_md_filter_multi(drug_candidates: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not drug_candidates:
        return None
    families = [guess_drug_family(d) for d in drug_candidates]
    families = [f for f in families if f]
    if not families:
        return None
    return {"drug_family": {"$in": families}}


def build_where_document(section_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    if not section_hint:
        return None
    can = canonical_section_name(section_hint)
    return {"section_canonical": {"$eq": can}}


def build_where_document_drug(drug_candidates: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not drug_candidates:
        return None
    families = [guess_drug_family(d) for d in drug_candidates]
    families = [f for f in families if f]
    if not families:
        return None
    regexes = [re.compile(rf"\b{re.escape(f)}\b", re.I) for f in families]
    return {"$or": [{"drug_family": {"$regex": r.pattern}} for r in regexes]}


def detect_drug_candidates(vs: Chroma, query: str, topn: int = 5) -> List[str]:
    q = normalize_drug_name(query)
    if not q:
        return []
    try:
        meta = vs._collection.get(include=["metadatas"], limit=100_000).get("metadatas", [])
    except Exception:
        return []

    drug_names = set()
    for m in meta:
        dn = m.get("drug_name") or m.get("drug_family")
        if dn:
            drug_names.add(str(dn))

    if not drug_names:
        return []

    fams = sorted({guess_drug_family(d) for d in drug_names if d})
    candidates = difflib.get_close_matches(q, fams, n=topn, cutoff=0.87)
    out = []
    for c in candidates:
        for d in drug_names:
            if guess_drug_family(d) == c:
                out.append(d)
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq[:topn]


# ---------------------------------------------------------------------
# Múltiples consultas (reescritura)
# ---------------------------------------------------------------------


def expand_queries(
    q: str,
    section_hint: Optional[str],
    llm=None,
    max_rewrites: int = 3,
) -> List[str]:
    """
    Genera múltiples consultas a partir de la consulta base:
    - incluye la consulta original,
    - opcionalmente agrega reformulaciones con LLM,
    - opcionalmente antepone sinónimos de sección.
    """
    out: List[str] = [q]

    # 1) Reformulaciones con LLM
    if llm is not None and max_rewrites > 0:
        prompt = (
            "Reformulá la siguiente pregunta en español generando hasta "
            f"{max_rewrites} variantes equivalentes.\n"
            "No cambies el significado ni agregues información nueva.\n"
            "Devolvé cada variante en una línea separada, sin numerarlas.\n\n"
            f"Pregunta original: {q!r}"
        )
        try:
            resp = llm.invoke(prompt)
            # OllamaLLM devuelve normalmente un string
            raw = str(resp)
            candidates = [
                line.strip(" •-\t")
                for line in raw.splitlines()
                if line.strip()
            ]
            out.extend(candidates[:max_rewrites])
        except Exception:
            # si falla la reescritura, seguimos solo con la consulta original
            pass

    # 2) Prefijos con sinónimos de sección
    syns = synonyms_for(section_hint)
    if syns:
        out += [f"{s} {q}" for s in syns][:5]

    # 3) Deduplicar preservando orden
    seen, uniq = set(), []
    for s in out:
        s = s.strip()
        if not s:
            continue
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def similarity_search_safe(
    vs: Chroma,
    query: str,
    k: int,
    md_filter: Optional[Dict[str, Any]],
    where_doc: Optional[Dict[str, Any]],
) -> List[Tuple[Document, float]]:
    kwargs: Dict[str, Any] = {"k": max(k, 10)}
    if md_filter:
        kwargs["filter"] = md_filter
    if where_doc:
        kwargs["where_document"] = where_doc
    try:
        hits = vs.similarity_search_with_score(query, **kwargs)
        if not hits and ("filter" in kwargs or "where_document" in kwargs):
            kwargs.pop("filter", None)
            kwargs.pop("where_document", None)
            hits = vs.similarity_search_with_score(query, **kwargs)
        return hits
    except Exception:
        kwargs.pop("where_document", None)
        try:
            return vs.similarity_search_with_score(query, **kwargs)
        except Exception:
            return []


# ---------------------------------------------------------------------
# RRF merge + vecinos
# ---------------------------------------------------------------------


def rrf_merge(ranked_lists: List[List[Tuple[Document, float]]], k: int) -> List[Tuple[Document, float]]:
    RRF_K = 60
    scores: Dict[str, float] = defaultdict(float)
    docs: Dict[str, Document] = {}

    for l in ranked_lists:
        for rank, (doc, _score) in enumerate(l, start=1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            docs[doc_id] = doc
            scores[doc_id] += 1.0 / (RRF_K + rank)

    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(docs[doc_id], score) for doc_id, score in items]


def get_neighbors(vs: Chroma, doc: Document, neighbors: int = 1) -> List[Document]:
    if neighbors <= 0:
        return []
    md = doc.metadata or {}
    doc_id = md.get("doc_id")
    if not doc_id:
        return []

    try:
        res = vs._collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["metadatas", "documents"],
            limit=10_000,
        )
        metadatas = res.get("metadatas", [])
        documents = res.get("documents", [])
    except Exception:
        return []

    chunks = []
    for m, t in zip(metadatas, documents):
        ci = m.get("chunk_index", 0)
        chunks.append((ci, Document(page_content=t, metadata=m)))

    chunks.sort(key=lambda x: x[0])
    idx = None
    for i, (_, d) in enumerate(chunks):
        if d.metadata.get("chunk_index") == md.get("chunk_index"):
            idx = i
            break
    if idx is None:
        return []

    out = []
    for offset in range(1, neighbors + 1):
        if idx - offset >= 0:
            out.append(chunks[idx - offset][1])
        if idx + offset < len(chunks):
            out.append(chunks[idx + offset][1])
    return out


def attach_neighbors(
    vs: Chroma,
    docs_scores: List[Tuple[Document, float]],
    neighbors: int,
) -> List[Tuple[Document, float]]:
    if neighbors <= 0:
        return docs_scores

    new_docs: List[Tuple[Document, float]] = []
    seen_ids = set()

    for doc, score in docs_scores:
        doc_id = doc.metadata.get("id") or doc.page_content[:50]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            new_docs.append((doc, score))
        for nb in get_neighbors(vs, doc, neighbors):
            nb_id = nb.metadata.get("id") or nb.page_content[:50]
            if nb_id not in seen_ids:
                seen_ids.add(nb_id)
                new_docs.append((nb, score * 0.9))

    new_docs.sort(key=lambda x: x[1], reverse=True)
    return new_docs


# ---------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------


def keyword_score(doc: Document, query: str) -> float:
    text = (doc.page_content or "").lower()
    q = (query or "").lower()
    tokens_q = [t for t in re.split(r"\W+", q) if len(t) > 3]
    if not tokens_q:
        return 0.0
    score = 0.0
    for t in tokens_q:
        count = text.count(t)
        if count:
            score += 1.0 + 0.1 * (count - 1)
    return score


def rerank_keyword(
    docs_scores: List[Tuple[Document, float]],
    query: str,
) -> List[Tuple[Document, float]]:
    if not docs_scores:
        return docs_scores
    out = []
    for doc, base_score in docs_scores:
        kw = keyword_score(doc, query)
        out.append((doc, base_score + 0.5 * kw))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def make_llm(backend: str, model: str, temperature: float = 0.0):
    """
    Igual que en tu 4-retrieve.py original: usa OllamaLLM (no ChatOllama),
    con num_ctx grande y num_predict corto para respuestas concisas.
    """
    if backend == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model,
            temperature=temperature,
            num_ctx=16384,
            num_predict=256,
        )
    raise ValueError(f"Backend LLM no soportado: {backend}")


def rerank_cross(
    docs_scores: List[Tuple[Document, float]],
    query: str,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> List[Tuple[Document, float]]:
    if not docs_scores:
        return docs_scores
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except Exception:
        return docs_scores

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    pairs = []
    for doc, _ in docs_scores:
        text = doc.page_content or ""
        pairs.append((query, text))

    with torch.no_grad():
        inputs = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()

    out = []
    for (doc, _), sc in zip(docs_scores, scores):
        out.append((doc, float(sc)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


def make_embeddings(provider: str, openai_model: str, e5_model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Definí OPENAI_API_KEY para usar provider=openai.")
        return OpenAIEmbeddings(model=openai_model)

    if provider == "e5":
        from sentence_transformers import SentenceTransformer

        class E5Emb:
            def __init__(self, name: str):
                self.model = SentenceTransformer(name, device="cpu")

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                texts2 = [f"passage: {t}" for t in texts]
                return self.model.encode(
                    texts2,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).tolist()

            def embed_query(self, text: str) -> List[float]:
                q = f"query: {text}"
                return self.model.encode(
                    [q],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0].tolist()

        return E5Emb(e5_model)

    raise ValueError("provider debe ser 'openai' o 'e5'.")


# ---------------------------------------------------------------------
# Construcción de respuesta con LLM (prompt “foco en la pregunta”)
# ---------------------------------------------------------------------


def build_answer(
    llm,
    question: str,
    section_hint: Optional[str],  # no se usa en el prompt, pero se mantiene en la firma
    docs: Sequence[Document],
) -> str:
    """
    Construye la respuesta usando SOLO el contexto de los docs recuperados.

    Usa el prompt que definiste:
    - foco en UNA pregunta,
    - solo contexto,
    - si no hay info: "No hay información suficiente..."
    """

    # Armamos contexto numerado [n]
    context_parts = []
    for i, d in enumerate(docs, 1):
        text = (d.page_content or "").strip()
        if not text:
            continue
        context_parts.append(f"[{i}] {text}")
    context = "\n\n".join(context_parts) if context_parts else "(sin contexto disponible)"

    prompt = f"""
Sos un asistente de búsqueda en prospectos médicos. 
Tu única tarea es responder exactamente la pregunta indicada abajo, basándote 
EXCLUSIVAMENTE en el texto de los fragmentos numerados del CONTEXTO.

NO inventes, NO cambies el tema, y NO respondas a otra pregunta diferente.

PREGUNTA A RESPONDER (foco obligatorio):
{question}

CONTEXTO:
{context}

INSTRUCCIONES:
1. Leé la PREGUNTA y mantenela como único foco.
2. Si encontrás información en el CONTEXTO que responda directa o parcialmente a esa pregunta, resumila en 1–3 oraciones.
3. Si el CONTEXTO no tiene información relacionada con esa pregunta específica, respondé literalmente:
   "No hay información suficiente en el contexto para responder con precisión."
4. No respondas sobre otros temas ni uses conocimiento externo.
5. Citá los fragmentos usados con Referencias: [n].
""".strip()

    resp = llm.invoke(prompt)
    # OllamaLLM devuelve string
    answer_text = str(resp)
    return answer_text.strip()


# ---------------------------------------------------------------------
# Pretty-print resultados (chunks)
# ---------------------------------------------------------------------


def pretty_print_docs_scores(
    docs_scores: List[Tuple[Document, float]],
    max_docs: int = 10,
) -> None:
    if not docs_scores:
        emit("(sin resultados)")
        return

    for i, (doc, score) in enumerate(docs_scores[:max_docs], 1):
        md = doc.metadata or {}
        sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
        drug = md.get("drug_name") or ""
        doc_name = md.get("doc_name") or ""
        chunk_id = md.get("id")
        header = f"[{i}] score={score:.4f} | droga={drug} | sección={sec} | doc={doc_name}"
        if chunk_id:
            header += f" | id={chunk_id}"
        emit(header)
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        emit("  " + snippet)
        emit("-" * 80)


# ---------------------------------------------------------------------
# Post-filter por familia
# ---------------------------------------------------------------------


def post_filter_by_drug_family(
    docs_scores: List[Tuple[Document, float]],
    drug_candidates: Optional[List[str]],
) -> List[Tuple[Document, float]]:
    if not drug_candidates:
        return docs_scores
    families = {guess_drug_family(d) for d in drug_candidates if d}
    if not families:
        return docs_scores
    out: List[Tuple[Document, float]] = []
    for doc, score in docs_scores:
        fam = guess_drug_family(doc.metadata.get("drug_name") or "")
        if fam in families:
            out.append((doc, score))
    return out or docs_scores


def infer_majority_drug(docs: Sequence[Document]) -> Optional[str]:
    if not docs:
        return None
    fams = []
    for d in docs:
        fam = guess_drug_family(d.metadata.get("drug_name") or "")
        if fam:
            fams.append(fam)
    if not fams:
        return None
    c = Counter(fams)
    fam, cnt = c.most_common(1)[0]
    if cnt < max(3, len(docs) // 3):
        return None
    return fam


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Q&A sobre prospectos médicos (versión multi-query)."
    )
    ap.add_argument(
        "-q",
        "--query",
        required=True,
        help="Pregunta del usuario en lenguaje natural",
    )

    # Chroma
    ap.add_argument(
        "-p",
        "--persist-dir",
        default="chroma/prospectos",
        help="Directorio de Chroma (default: chroma/prospectos)",
    )
    ap.add_argument(
        "-c",
        "--collection",
        default="prospectos",
        help="Colección (default: prospectos)",
    )

    # Embeddings
    ap.add_argument("--provider", choices=["openai", "e5"], default="e5")
    ap.add_argument("--openai-model", default="text-embedding-3-small")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base")

    # Recuperación
    ap.add_argument("-k", "--topk", type=int, default=12)
    ap.add_argument("--neighbors", type=int, default=1)
    ap.add_argument(
        "--multi-query-rewrites",
        type=int,
        default=0,
        help="Cantidad de reformulaciones de la consulta base (0 = desactivado)",
    )
    ap.add_argument(
        "--rerank",
        choices=["none", "keyword", "cross"],
        default="keyword",
    )
    ap.add_argument(
        "--rerank-model", default="BAAI/bge-reranker-v2-m3"
    )

    # Respuesta
    ap.add_argument(
        "--answer",
        action="store_true",
        help="Si se pasa, genera respuesta con LLM",
    )
    ap.add_argument(
        "--llm-backend", default="ollama", help="Backend LLM (default: ollama)"
    )
    ap.add_argument(
        "--llm-model", default="llama3.1", help="Modelo LLM para respuesta (como en 4-retrieve.py)"
    )
    ap.add_argument(
        "--llm-temperature", type=float, default=0.0, help="Temperatura del LLM (default 0.0)"
    )

    # Output
    ap.add_argument(
        "--output",
        choices=["full", "answer"],
        default="full",
        help="full: chunks + respuesta; answer: SOLO respuesta del LLM.",
    )
    ap.add_argument(
        "--debug", action="store_true", help="Imprime información de debug"
    )

    args = ap.parse_args(argv)

    # ---- Abrir Chroma + embeddings ----
    if not os.path.isdir(args.persist_dir):
        emit(f"⚠️ No existe el directorio de Chroma: {args.persist_dir}")
        return 1

    emb = make_embeddings(args.provider, args.openai_model, args.e5_model)
    vs = Chroma(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_function=emb,
    )

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

    # ---- Detección de droga y sección ----
    query = (args.query or "").strip()
    drug_candidates = detect_drug_candidates(vs, query)
    sec_hint = auto_detect_section_hint(query)

    if args.debug:
        emit(f"→ Drogas detectadas (familia): {drug_candidates or 'ninguna'} | section_hint={sec_hint}")

    md_filter = build_md_filter_multi(drug_candidates)
    where_sec = build_where_document(sec_hint)
    where_drug = build_where_document_drug(drug_candidates)
    where_and = (
        {"$and": [wd for wd in (where_sec, where_drug) if wd]}
        if (where_sec or where_drug)
        else None
    )

    # LLM para reescritura y respuesta (mismo modelo/config)
    rewrite_llm = None
    if getattr(args, "multi_query_rewrites", 0) > 0:
        try:
            rewrite_llm = make_llm(args.llm_backend, args.llm_model, temperature=args.llm_temperature)
            if args.debug:
                emit(f"→ Reescritura activada: {args.multi_query_rewrites} variantes")
        except Exception as e:
            rewrite_llm = None
            if args.debug:
                emit(f"⚠️ No se pudo inicializar LLM para reescritura: {type(e).__name__}: {e}")

    # -------------------------------------------------
    # 1ª pasada: búsqueda base + múltiples consultas
    # -------------------------------------------------
    queries = expand_queries(
        query,
        sec_hint,
        llm=rewrite_llm,
        max_rewrites=getattr(args, "multi_query_rewrites", 0),
    )

    if args.debug:
        emit("→ Consultas generadas (multi-query):")
        for i, qi in enumerate(queries):
            emit(f"   Q{i}: {qi}")

    all_hits: List[List[Tuple[Document, float]]] = []

    # búsqueda base sin filtros, como fallback
    try:
        base_hits = vs.similarity_search_with_score(query, k=max(args.topk, 10))
    except Exception:
        base_hits = []
    all_hits.append(base_hits)

    for qi in queries:
        hits = similarity_search_safe(
            vs,
            qi,
            args.topk,
            md_filter=md_filter,
            where_doc=where_and,
        )
        all_hits.append(hits)

    docs_scores = rrf_merge(all_hits, args.topk)

    # Post-filter por familia
    docs_scores = post_filter_by_drug_family(docs_scores, drug_candidates)

    # 2ª pasada opcional: mayoría de droga si no había candidatos
    if not drug_candidates and docs_scores:
        maj = infer_majority_drug([d for d, _ in docs_scores])
        if args.debug:
            emit(f"→ Mayoría por resultados: {maj or 'ninguna clara'}")
        if maj:
            md_filter2 = build_md_filter_multi([maj])
            where_drug2 = build_where_document_drug([maj])
            where_and2 = (
                {"$and": [wd for wd in (where_sec, where_drug2) if wd]}
                if (where_sec or where_drug2)
                else None
            )
            all_hits = []
            try:
                base_hits2 = vs.similarity_search_with_score(query, k=max(args.topk, 10))
            except Exception:
                base_hits2 = []
            all_hits.append(base_hits2)
            for qi in queries:
                hits = similarity_search_safe(
                    vs,
                    qi,
                    args.topk,
                    md_filter=md_filter2,
                    where_doc=where_and2,
                )
                all_hits.append(hits)
            docs_scores = rrf_merge(all_hits, args.topk)

    # Vecinos
    docs_scores = attach_neighbors(vs, docs_scores, args.neighbors)

    # Re-ranking
    if args.rerank == "keyword":
        docs_scores = rerank_keyword(docs_scores, query)
    elif args.rerank == "cross":
        docs_scores = rerank_cross(docs_scores, query, model_name=args.rerank_model)

    # Output full: mostrar chunks recuperados (stderr)
    if args.output == "full":
        pretty_print_docs_scores(docs_scores, max_docs=args.topk)

    # Respuesta con LLM
    if args.answer and docs_scores:
        # reutilizamos el mismo modelo (con mismos params) para la respuesta
        llm = make_llm(args.llm_backend, args.llm_model, temperature=args.llm_temperature)
        try:
            ans = build_answer(
                llm,
                query,
                sec_hint,
                [d for d, _ in docs_scores[: args.topk]],
            )
            if args.output == "answer":
                # SOLO respuesta limpia, para piping / evaluación
                print(ans.strip())
            else:
                emit("\n===================")
                emit("PREGUNTA DEL USUARIO")
                emit("===================")
                emit(query)
                emit("\n===================")
                emit("RESPUESTA DEL SISTEMA")
                emit("===================")
                emit(ans.strip())
                emit("\n")
        except Exception as e:
            if args.output == "answer":
                print("No se pudo generar respuesta con el LLM.")
            emit(f"⚠️ Error generando respuesta con LLM: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
