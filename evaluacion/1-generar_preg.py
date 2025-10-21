#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generar 1 pregunta por chunk usando OLLAMA vía HTTP (sin LangChain).

IO por defecto:
  IN : /Users/vivi/qbya-project/qbya/out_chunks/chunks.jsonl
  OUT: /Users/vivi/qbya-project/qbya/evaluacion/questions.jsonl
  CSV: /Users/vivi/qbya-project/qbya/evaluacion/questions.csv

Requisitos:
  (.qbya) pip install -U requests pandas

Debe estar corriendo el servidor de Ollama y el modelo descargado:
  ollama serve
  ollama pull <modelo>   # ej: ollama pull llama3:8b-instruct

Ejemplo:
  (.qbya) python 1-generar_preg.py \
    --model llama3:8b-instruct \
    --max-chunks 200 --min-chars 80 --dedup
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List

import requests

try:
    import pandas as pd
except Exception:
    pd = None


# -------------------------
# Paths por defecto (tu estructura)
# -------------------------
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
DEFAULT_IN  = BASE_DIR / "out_chunks" / "chunks.jsonl"
EVAL_DIR    = BASE_DIR / "evaluacion"
DEFAULT_OUT = EVAL_DIR / "1-questions.jsonl"
DEFAULT_CSV = EVAL_DIR / "1-questions.csv"


# -------------------------
# Lectura de JSONL
# -------------------------
def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_context_and_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    context = obj.get("page_content") or obj.get("text") or ""
    metadata = obj.get("metadata") or {}
    chunk_id = obj.get("id") or metadata.get("id")
    for k in ("drug", "section", "file", "source", "doc_id"):
        if k in obj and k not in metadata:
            metadata[k] = obj[k]
    return {"context": context, "chunk_id": chunk_id, "metadata": metadata}


# -------------------------
# Prompt (tu versión, ajustado a salida plana)
# -------------------------
def build_prompt(context: str) -> str:
    return f"""<instructions>
Here is some context:
<context>
{context}
</context>
<role>You are a teacher creating a quiz from a given context.</role>
<task>
Your task is to generate 1 question that can be answered using the provided context, following these rules:

<rules>
1. The question should make sense to humans even when read without the given context.
2. The question should be fully answered from the given context.
3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
4. The answer to the question should not contain any links.
5. The question should be of moderate difficulty.
6. The question must be reasonable and must be understood and responded by humans.
7. Do not use phrases like 'provided context', etc. in the question.
8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
9. The question should not contain more than 10 words, make use of abbreviations wherever possible.
10. The question must explicitly include a medication mention from the context: either the commercial brand name, the generic/active substance, or another clear term that identifies the drug (e.g., "Ozempic", "semaglutide", "acetaminophen", etc.).
</rules>

To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

Output only the generated question with a "?" at the end, no other text or characters.
</task>
</instructions>""".strip()


# -------------------------
# Cliente Ollama (HTTP)
# -------------------------
def ollama_tags(base_url: str) -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
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


def ollama_chat(
    base_url: str,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float = 0.2,
    retries: int = 1,
    sleep_sec: float = 0.4,
) -> str:
    """
    Usa /api/chat. Maneja 404 con sugerencia de modelos disponibles.
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
        "options": {
            "temperature": float(temperature),
        },
    }

    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 404:
                avail = ollama_tags(base_url)
                msg = (
                    f"[ERROR] Modelo de Ollama no disponible: '{model}'.\n"
                    f"Modelos instalados: {', '.join(avail) if avail else '(ninguno listado)'}\n"
                    f"Sugerencia: `ollama pull {model}` o usá uno de la lista.\n"
                )
                raise RuntimeError(msg)
            r.raise_for_status()
            data = r.json()
            # Respuesta consolidada de /api/chat
            content = ""
            for choice in data.get("message", {}).get("content", ""):
                # a veces content es string directo; a veces char por char (stream desactivado => debería ser string)
                pass
            content = data.get("message", {}).get("content", "")
            return (content or "").strip()
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(sleep_sec)

    return ""


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Ruta de chunks JSONL")
    ap.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="Ruta de salida JSONL")
    ap.add_argument("--csv", dest="csv_path", default=str(DEFAULT_CSV), help="Ruta CSV opcional")
    ap.add_argument("--model", default="llama3:8b-instruct",
                    help="Modelo de Ollama (ver `ollama list`), ej: llama3:8b-instruct, qwen2.5:7b-instruct")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434",
                    help="Base URL de Ollama (p.ej., http://127.0.0.1:11434)")
    ap.add_argument("--system", default="Eres un generador de preguntas conciso y preciso.",
                    help="System prompt opcional")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-chunks", type=int, default=None, help="Límite de chunks a procesar")
    ap.add_argument("--min-chars", type=int, default=50, help="Descartar chunks muy cortos")
    ap.add_argument("--dedup", action="store_true", help="Eliminar preguntas duplicadas")
    ap.add_argument("--stride", type=int, default=1,
                help="Procesar 1 de cada N chunks (ej.: 15 para muestreo más amplio)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    csv_path = Path(args.csv_path) if args.csv_path else None

    # Asegurar carpeta de evaluación
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Chequeo rápido de conectividad/modelo
    try:
        _ = ollama_tags(args.ollama_base_url)
    except Exception:
        sys.stderr.write("[ERROR] No puedo conectar con Ollama. ¿Está corriendo `ollama serve`?\n")
        sys.exit(1)

    results = []
    seen_questions = set()
    processed = 0

    for idx, obj in enumerate(read_jsonl(in_path)):
    # Salto de N en N (p. ej., 15)
        if args.stride > 1 and (idx % args.stride != 0):
            continue
        data = extract_context_and_meta(obj)
        context = data["context"]
        if not context or len(context) < args.min_chars:
            continue

        prompt = build_prompt(context)
        try:
            q = ollama_chat(
                base_url=args.ollama_base_url,
                model=args.model,
                system_prompt=args.system,
                user_prompt=prompt,
                temperature=args.temperature,
                retries=1,
                sleep_sec=0.4,
            )
        except Exception as e:
            # 404 o conexión: mostramos y seguimos al próximo chunk
            sys.stderr.write(str(e) + "\n")
            break  # si el modelo falta, no tiene sentido seguir

        if not q:
            continue

        # Normalización mínima
        q = q.replace("\n", " ").strip()
        if not q.endswith("?"):
            q = (q + "?").strip()

        if args.dedup:
            key = q.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)

        preview = context.strip().replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240] + "…"

        results.append(
            {
                "chunk_id": data["chunk_id"],
                "question": q,
                "context_preview": preview,
                "metadata": data["metadata"],
            }
        )

        processed += 1
        if args.max_chunks and processed >= args.max_chunks:
            break

    # Guardar JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Guardar CSV (opcional)
    if csv_path:
        if pd is None:
            sys.stderr.write("[WARN] pandas no instalado; omitiendo CSV\n")
        else:
            flat = []
            for r in results:
                row = {
                    "chunk_id": r["chunk_id"],
                    "question": r["question"],
                    "context_preview": r["context_preview"],
                }
                md = r.get("metadata") or {}
                for k in ("drug", "section", "file", "source", "doc_id"):
                    if k in md:
                        row[k] = md[k]
                flat.append(row)
            pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✓ Generadas {len(results)} preguntas -> {out_path}")
    if csv_path:
        print(f"✓ CSV -> {csv_path}")


if __name__ == "__main__":
    main()
