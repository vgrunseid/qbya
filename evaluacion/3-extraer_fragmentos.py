#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3-extraer_fragmentos.py — Extraer oraciones relevantes del contexto para cada pregunta.

Entradas por defecto:
  - chunks.jsonl : /Users/vivi/qbya-project/qbya/out_chunks/chunks.jsonl
  - answers.csv  : /Users/vivi/qbya-project/qbya/evaluacion/answers.csv
                   (si no existe, usa questions.csv)

Salidas por defecto:
  - JSONL: /Users/vivi/qbya-project/qbya/evaluacion/relevant_fragments.jsonl
  - CSV  : /Users/vivi/qbya-project/qbya/evaluacion/relevant_fragments.csv

Características:
  - Usa Ollama vía HTTP, con timeouts (no se cuelga).
  - Verbose y progreso cada N filas (--log-every).
  - Limita el tamaño del contexto (--max-context-chars).
  - Con --show-chunks imprime SOLO los IDs de los chunks usados por contexto (sin contenido).
  - Con --doc-window K (y --scope document) toma a lo sumo K chunks del doc, centrados en el base.

Requisitos:
  (.qbya) pip install -U pandas requests
  ollama pull llama3.1:latest   # o el tag que uses
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd

# ---- Paths por defecto
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
CHUNKS_JSONL = BASE_DIR / "out_chunks" / "chunks.jsonl"
EVAL_DIR = BASE_DIR / "evaluacion"
ANSWERS_CSV = EVAL_DIR / "2-answers.csv"
QUESTIONS_CSV = EVAL_DIR / "1-questions.csv"
OUT_JSONL = EVAL_DIR / "3-relevant_fragments.jsonl"
OUT_CSV = EVAL_DIR / "3-relevant_fragments.csv"

# ---- Prompt (con llaves escapadas)
SOURCE_PROMPT = """<instructions>
<role>You are an experienced QA Engineer for building large language model applications.</role>
<task>
Your task is to extract the relevant sentences from the given context that can potentially help answer the following question.

<rules>
1) You are NOT allowed to change, paraphrase or complete the sentences. Copy them verbatim from the context.
2) Select only the minimal set of sentences needed to support the answer.
3) If no sentence is relevant, return an empty list.
4) Output JSON ONLY with this exact schema:
   {{\"sentences\": [\"<sentence 1>\", \"<sentence 2>\", ...]}}
   (no markdown, no extra keys).
</rules>

Here is the context:
<context>
{full_context}
</context>

Here is the question:
<question>
{question}
</question>
</task>
</instructions>""".strip()


# ---------- Utils básicos ----------
def read_chunks(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    """
    Devuelve:
      - chunk_map: id_chunk -> objeto chunk
      - doc_map:   doc_key  -> lista de (chunk_id, texto) en orden
    'doc_key' se arma con metadata.doc_id o metadata.source o metadata.file.
    """
    chunk_map: Dict[str, Dict[str, Any]] = {}
    doc_map: Dict[str, List[Tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id") or (obj.get("metadata") or {}).get("id")
            if not cid:
                continue
            chunk_map[cid] = obj
            md = obj.get("metadata") or {}
            doc_key = md.get("doc_id") or md.get("source") or md.get("file") or f"__chunk__:{cid}"
            text = (obj.get("page_content") or obj.get("text") or "").strip()
            if text:
                doc_map.setdefault(doc_key, []).append((cid, text))
    return chunk_map, doc_map


def get_text_from_chunk(obj: Dict[str, Any]) -> str:
    return (obj.get("page_content") or obj.get("text") or "").strip()


def ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def resolve_model_tag(requested: str, available: List[str]) -> str:
    """
    Si el usuario pasa 'name' y existe 'name:latest', usamos ese.
    Si pasa un tag exacto que existe, lo dejamos tal cual.
    """
    if requested in available:
        return requested
    if ":" not in requested:
        candidate = f"{requested}:latest"
        if candidate in available:
            return candidate
    return requested


# ---------- Cliente Ollama HTTP con timeout ----------
def ollama_list_models(base_url: str, timeout: float = 10.0) -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        names = []
        for m in data.get("models", []):
            n = m.get("model") or m.get("name")
            if n:
                names.append(n)
        return names
    except Exception:
        return []


def ollama_chat_json(
    base_url: str,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float = 0.0,
    timeout_connect: float = 10.0,
    timeout_read: float = 120.0,
) -> dict:
    """
    Llama /api/chat de Ollama con stream=False y timeouts.
    Devuelve el JSON crudo de respuesta (dict).
    """
    url = f"{base_url}/api/chat"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    r = requests.post(url, json=payload, timeout=(timeout_connect, timeout_read))
    if r.status_code == 404:
        avail = ollama_list_models(base_url)
        raise RuntimeError(
            f"Modelo de Ollama '{model}' no disponible (404).\n"
            f"Modelos instalados: {', '.join(avail) if avail else '(ninguno)'}\n"
            f"Sugerencia: `ollama pull {model}` o usá uno de la lista."
        )
    r.raise_for_status()
    return r.json()


def extract_sentences_from_response(resp: dict) -> List[str]:
    """Parsea respuesta de /api/chat esperando JSON embebido en el content."""
    content = resp.get("message", {}).get("content", "")
    content = (content or "").strip()
    # 1) intento: parsear como JSON directo
    try:
        data = json.loads(content)
        s = data.get("sentences", [])
        if isinstance(s, list):
            return [x.strip() for x in s if isinstance(x, str) and x.strip()]
    except Exception:
        pass
    # 2) buscar primer bloque {...}
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            s = data.get("sentences", [])
            if isinstance(s, list):
                return [x.strip() for x in s if isinstance(x, str) and x.strip()]
        except Exception:
            pass
    # 3) fallback: una oración por línea
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    cleaned = [ln.lstrip("-•* ").strip() for ln in lines]
    return cleaned


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(CHUNKS_JSONL))
    ap.add_argument("--answers", default=str(ANSWERS_CSV))
    ap.add_argument("--questions", default=str(QUESTIONS_CSV))
    ap.add_argument("--out-jsonl", default=str(OUT_JSONL))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--scope", choices=["chunk", "document"], default="chunk")
    ap.add_argument("--doc-window", type=int, default=0,
                    help="Si >0 y scope=document, usa a lo sumo K chunks centrados en el base.")
    ap.add_argument("--max-context-chars", type=int, default=12000,
                    help="Trunca el contexto a este máximo de caracteres para evitar cuelgues.")
    ap.add_argument("--timeout-connect", type=float, default=10.0)
    ap.add_argument("--timeout-read", type=float, default=180.0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=5, help="Log de progreso cada N filas.")
    # Mostrar solo IDs de chunks usados:
    ap.add_argument("--show-chunks", action="store_true",
                    help="Imprime solo los IDs de los chunks usados para cada contexto (sin contenido).")
    args = ap.parse_args()

    # Asegurar carpetas de salida
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Cargar base (answers > questions)
    answers_path = Path(args.answers)
    if answers_path.exists():
        df = pd.read_csv(answers_path)
        required = {"chunk_id", "question"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"{answers_path} debe tener columnas {required}. Columnas: {df.columns.tolist()}")
    else:
        qpath = Path(args.questions)
        if not qpath.exists():
            raise RuntimeError(f"No se encontró ni {answers_path} ni {qpath}")
        df = pd.read_csv(qpath)
        required = {"chunk_id", "question"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"{qpath} debe tener columnas {required}. Columnas: {df.columns.tolist()}")

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # Cargar chunks
    chunk_map, doc_map = read_chunks(Path(args.chunks))
    if not chunk_map:
        raise RuntimeError(f"No se cargaron chunks desde {args.chunks}")

    # Chequeo Ollama y resolución de tag
    models = ollama_list_models(args.ollama_base_url)
    resolved_model = resolve_model_tag(args.model, models)
    if resolved_model != args.model:
        print(f"[INFO] Usando '{resolved_model}' en lugar de '{args.model}'", flush=True)
        args.model = resolved_model
    if args.model not in models:
        sys.stderr.write(
            f"[WARN] El modelo '{args.model}' no aparece en ollama list.\n"
            f"Modelos: {', '.join(models) if models else '(ninguno)'}\n"
            f"Intento de todos modos…\n"
        )

    rows_out = []
    total = len(df)
    t0 = time.time()

    for i, row in df.iterrows():
        chunk_id = str(row.get("chunk_id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not chunk_id or not question:
            sys.stderr.write(f"[WARN] Fila {i}: falta chunk_id o question. Salto.\n")
            continue

        ch = chunk_map.get(chunk_id)
        if not ch:
            sys.stderr.write(f"[WARN] Fila {i}: chunk_id={chunk_id} no existe en {args.chunks}. Salto.\n")
            continue

        if args.scope == "chunk":
            included_chunk_ids = [chunk_id]
            full_context = get_text_from_chunk(ch)
        else:
            md = ch.get("metadata") or {}
            doc_key = md.get("doc_id") or md.get("source") or md.get("file") or f"__chunk__:{chunk_id}"
            pairs = doc_map.get(doc_key, [])
            if not pairs:
                included_chunk_ids = [chunk_id]
                full_context = get_text_from_chunk(ch)
            else:
                # ventana centrada si se pidió
                if args.doc_window and args.doc_window > 0:
                    # encontrar índice del base
                    base_idx = next((idx for idx, (cid, _) in enumerate(pairs) if cid == chunk_id), None)
                    if base_idx is None:
                        base_idx = 0
                    K = max(1, int(args.doc_window))
                    left = (K - 1) // 2
                    right = K - 1 - left
                    start = max(0, base_idx - left)
                    end = min(len(pairs), base_idx + right + 1)
                    window_pairs = pairs[start:end]
                    included_chunk_ids = [cid for cid, _ in window_pairs]
                    parts = [txt for _, txt in window_pairs]
                else:
                    included_chunk_ids = [cid for cid, _ in pairs]
                    parts = [txt for _, txt in pairs]
                full_context = "\n\n".join(parts)

        if not full_context.strip():
            sys.stderr.write(f"[WARN] Fila {i}: contexto vacío. Salto.\n")
            continue

        # Trunca contexto para evitar cuelgues
        full_context_capped = ellipsis(full_context, args.max_context_chars)

        # Mostrar solo IDs de chunks usados (sin contenido)
        if args.show_chunks:
            print(f"[CTX] fila={i}  scope={args.scope}  chunk_id_base={chunk_id}")
            print(f"[CHUNKS_INCLUDED] {included_chunk_ids}", flush=True)

        # Armar prompt
        user_prompt = SOURCE_PROMPT.format(full_context=full_context_capped, question=question)
        system_prompt = "You extract verbatim sentences from the context. Output JSON only."

        try:
            resp = ollama_chat_json(
                base_url=args.ollama_base_url,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                timeout_connect=args.timeout_connect,
                timeout_read=args.timeout_read,
            )
            sentences = extract_sentences_from_response(resp)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}): {e}\n")
            sentences = []

        rows_out.append({
            "chunk_id": chunk_id,
            "question": question,
            "answer": str(row.get("answer")) if "answer" in df.columns else None,
            "n_sentences": len(sentences),
            "sentences": sentences,                          # JSON array en JSONL
            "sentences_join": " || ".join(sentences),        # útil para CSV
        })

        if args.sleep > 0:
            time.sleep(args.sleep)
        if args.log_every and ((i + 1) % args.log_every == 0):
            elapsed = time.time() - t0
            print(f"[{i+1}/{total}] OK - t={elapsed:.1f}s", flush=True)

    # Guardar JSONL
    with Path(args.out_jsonl).open("w", encoding="utf-8") as f:
        for r in rows_out:
            j = dict(r)
            j.pop("sentences_join", None)
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # Guardar CSV
    pd.DataFrame(rows_out).to_csv(Path(args.out_csv), index=False, encoding="utf-8")

    print(f"✓ Extraídos fragmentos para {len(rows_out)} preguntas -> {args.out_jsonl}")
    print(f"✓ CSV -> {args.out_csv}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
