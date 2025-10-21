#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-evolucionar_preg.py — Reescribe (evoluciona) preguntas para robustecer el testset del RAG.

Entradas por defecto:
  - Preguntas/Respuestas:
      * answers.csv  : /Users/vivi/qbya-project/qbya/evaluacion/answers.csv
      * (fallback) questions.csv : /Users/vivi/qbya-project/qbya/evaluacion/questions.csv
  - Chunks (para metadata de droga si se usa --preserve-drug-mention):
      * /Users/vivi/qbya-project/qbya/out_chunks/chunks.jsonl

Salidas por defecto:
  - JSONL: /Users/vivi/qbya-project/qbya/evaluacion/evolved_questions.jsonl
  - CSV  : /Users/vivi/qbya-project/qbya/evaluacion/evolved_questions.csv

Características:
  - Ollama vía HTTP con timeouts y progreso.
  - Reescritura a forma indirecta/comprimida (7–10 palabras).
  - Opción --preserve-drug-mention: mantener la mención del medicamento (marca o genérico).
  - Validación de longitud y (si corresponde) de mención; reintento suave si no cumple.

Requisitos:
  (.qbya) pip install -U pandas requests
  ollama pull llama3.1:latest   # o el tag que uses
"""

import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd

# ---- Paths por defecto
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
EVAL_DIR = BASE_DIR / "evaluacion"
ANSWERS_CSV = EVAL_DIR / "2-answers.csv"
QUESTIONS_CSV = EVAL_DIR / "1-questions.csv"
CHUNKS_JSONL = BASE_DIR / "out_chunks" / "chunks.jsonl"
OUT_JSONL = EVAL_DIR / "4-evolved_questions.jsonl"
OUT_CSV = EVAL_DIR / "4-evolved_questions.csv"

# ---- Prompt base (llaves escapadas sólo si mostrásemos JSON literal en el prompt; aquí pedimos JSON en la salida)
QUESTION_EVOLVE_PROMPT = """<instructions>
<role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

<task>
Rewrite the following question to be more indirect and compressed. Must be in spanish

<rules>
1) Make it more indirect.
2) Make it shorter.
3) Use abbreviations if possible.
4) The question MUST have between 7 and 10 words.
5) End with a question mark "?". No extra punctuation.
6) Output JSON ONLY with this schema: {{"question": "<rewritten>"}}
</rules>

Original question:
<question>
{question}
</question>
{extra_constraints}
</task>
</instructions>""".strip()


# ---------- Utils básicos ----------
def ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def load_chunks_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Mapa chunk_id -> objeto chunk completo (para extraer metadata como drug/brand).
    """
    mp: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return mp
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id") or (obj.get("metadata") or {}).get("id")
            if cid:
                mp[cid] = obj
    return mp


def extract_med_terms(context_text: str, metadata: Dict[str, Any]) -> List[str]:
    """
    Devuelve posibles términos de medicamento (marca/genérico) desde metadata y texto.
    """
    terms = set()

    md = metadata or {}
    for k in ("drug", "brand", "brand_name", "generic", "active_ingredient", "medicine", "product"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            terms.add(v.strip())

    # heurística: capturar posibles nombres con mayúscula inicial (1-3 tokens)
    pattern = r'([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,2})(?:\s*(?:®|™))?'
    for m in re.finditer(pattern, context_text or ""):
        cand = m.group(1).strip()
        if len(cand) >= 3 and not cand.isupper():
            terms.add(cand)

    # hints de “contiene/ingrediente/etc.”
    hint = r'(?:contiene|composición|ingrediente|dosis|formulación)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9\- ]{2,40})'
    for m in re.finditer(hint, context_text or "", flags=re.IGNORECASE):
        terms.add(m.group(1).strip())

    uniq = []
    seen = set()
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)
    return uniq


def ensure_7_to_10_words(q: str) -> bool:
    tokens = [t for t in re.split(r"\s+", q.strip()) if t]
    return 7 <= len(tokens) <= 10


def contains_any_term(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)


# ---------- Cliente Ollama HTTP ----------
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
    temperature: float = 0.5,
    timeout_connect: float = 10.0,
    timeout_read: float = 120.0,
) -> dict:
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
            f"Sugerencia: `ollama pull {model}`."
        )
    r.raise_for_status()
    return r.json()


def parse_evolved_from_resp(resp: dict) -> str:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    content = content.strip()
    # 1) JSON directo
    try:
        data = json.loads(content)
        q = data.get("question", "")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except Exception:
        pass
    # 2) primer bloque { ... }
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            q = data.get("question", "")
            if isinstance(q, str) and q.strip():
                return q.strip()
        except Exception:
            pass
    # 3) fallback: tomar la línea más “pregunta-like”
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    for ln in lines:
        if ln.endswith("?"):
            return ln
    return content


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(ANSWERS_CSV))
    ap.add_argument("--questions", default=str(QUESTIONS_CSV))
    ap.add_argument("--chunks", default=str(CHUNKS_JSONL),
                    help="Necesario si usás --preserve-drug-mention para extraer términos desde metadata.")
    ap.add_argument("--out-jsonl", default=str(OUT_JSONL))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--print-evolved", action="store_true",
                    help="Muestra en consola 'original -> evolved'.")
    ap.add_argument("--preserve-drug-mention", action="store_true",
                    help="Intenta preservar una mención de la droga/marca en la pregunta reescrita.")
    ap.add_argument("--retry", type=int, default=1,
                    help="Reintentos si no cumple validaciones (longitud o mención).")
    args = ap.parse_args()

    # Asegurar carpetas
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Cargar base (answers > questions)
    df = None
    apath = Path(args.answers)
    if apath.exists():
        df = pd.read_csv(apath)
        required = {"chunk_id", "question"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"{apath} debe tener columnas {required}. Encontradas: {df.columns.tolist()}")
    else:
        qpath = Path(args.questions)
        if not qpath.exists():
            raise RuntimeError(f"No se encontró ni {apath} ni {qpath}")
        df = pd.read_csv(qpath)
        required = {"chunk_id", "question"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"{qpath} debe tener columnas {required}. Encontradas: {df.columns.tolist()}")

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # Cargar chunks para metadata (si vamos a preservar mención)
    chunk_meta_map: Dict[str, Dict[str, Any]] = {}
    if args.preserve_drug_mention:
        chunk_meta_map = load_chunks_metadata(Path(args.chunks))

    # Chequeo/modelos
    models = ollama_list_models(args.ollama_base_url)
    if args.model not in models and f"{args.model}:latest" in models:
        print(f"[INFO] Usando '{args.model}:latest' en lugar de '{args.model}'", flush=True)
        args.model = f"{args.model}:latest"
    elif args.model not in models:
        sys.stderr.write(f"[WARN] El modelo '{args.model}' no aparece en ollama list. Intento de todos modos…\n")

    out_rows = []
    total = len(df)
    t0 = time.time()

    for i, row in df.iterrows():
        chunk_id = str(row.get("chunk_id") or "").strip()
        original_q = str(row.get("question") or "").strip()
        if not chunk_id or not original_q:
            sys.stderr.write(f"[WARN] Fila {i}: falta chunk_id o question. Salto.\n")
            continue

        extra_constraints = ""
        terms: List[str] = []

        if args.preserve_drug_mention:
            # sacar terms desde metadata + (opcional) desde text del chunk si está presente
            md = (chunk_meta_map.get(chunk_id) or {}).get("metadata") or {}
            context_text = (chunk_meta_map.get(chunk_id) or {}).get("page_content") or (chunk_meta_map.get(chunk_id) or {}).get("text") or ""
            terms = extract_med_terms(context_text, md)
            if terms:
                terms_list = ", ".join(sorted(set(terms))[:6])
                extra_constraints = (
                    "\nAdditional constraint: The rewritten question MUST include at least one explicit medication mention "
                    f"from the context (brand or generic), e.g., {terms_list}."
                )
            else:
                # no terms found; aún así intentamos que la reescritura no elimine menciones si existieran
                extra_constraints = (
                    "\nAdditional constraint: If the original question contains a medication mention (brand or generic), "
                    "preserve an explicit mention in the rewritten question."
                )

        user_prompt = QUESTION_EVOLVE_PROMPT.format(
            question=original_q,
            extra_constraints=extra_constraints
        )

        system_prompt = "You rewrite questions following the rules and return JSON only."

        evolved = None
        attempts = 0
        last_err: Optional[Exception] = None

        while attempts <= args.retry:
            try:
                resp = ollama_chat_json(
                    base_url=args.ollama_base_url,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    timeout_connect=10.0,
                    timeout_read=120.0,
                )
                cand = parse_evolved_from_resp(resp).strip()
                # normalizaciones mínimas
                cand = re.sub(r"\s+", " ", cand)
                if not cand.endswith("?"):
                    cand = (cand + "?").strip()

                # validaciones
                ok_len = ensure_7_to_10_words(cand)
                ok_term = True
                if args.preserve_drug_mention and terms:
                    ok_term = contains_any_term(cand, terms)

                if ok_len and ok_term:
                    evolved = cand
                    break

                # si falla validación, agregamos recordatorio y reintentamos
                remind = []
                if not ok_len:
                    remind.append("Ensure 7–10 words exactly.")
                if args.preserve_drug_mention and terms and not ok_term:
                    remind.append("Include at least one of these medication mentions: " + ", ".join(sorted(set(terms))[:6]) + ".")
                if remind:
                    user_prompt = QUESTION_EVOLVE_PROMPT.format(
                        question=original_q,
                        extra_constraints="\nReminder: " + " ".join(remind)
                    )
            except Exception as e:
                last_err = e
            attempts += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        if evolved is None:
            if last_err:
                sys.stderr.write(f"[WARN] Fila {i}: fallo al reescribir. Error: {last_err}\n")
            # como fallback, intenta recortar/pulir la original
            evolved = original_q.strip()
            if not evolved.endswith("?"):
                evolved += "?"

        if args.print_evolved:
            print(f"- [{i+1}/{total}] {original_q}  -->  {evolved}", flush=True)

        out_rows.append({
            "chunk_id": chunk_id,
            "question_original": original_q,
            "question_evolved": evolved,
        })

        if args.log_every and ((i + 1) % args.log_every == 0):
            elapsed = time.time() - t0
            print(f"[{i+1}/{total}] OK - t={elapsed:.1f}s", flush=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Guardar JSONL
    with Path(args.out_jsonl).open("w", encoding="utf-8") as jf:
        for r in out_rows:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Guardar CSV
    pd.DataFrame(out_rows).to_csv(Path(args.out_csv), index=False, encoding="utf-8")

    print(f"✓ Evolucionadas {len(out_rows)} preguntas -> {args.out_jsonl}")
    print(f"✓ CSV -> {args.out_csv}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

