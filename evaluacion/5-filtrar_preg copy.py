#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5-filtrar_preg.py — Evaluar y filtrar preguntas por groundedness (1–5).

Entradas (en este orden de preferencia):
  - /Users/vivi/qbya-project/qbya/evaluacion/evolved_questions.csv   (si --use-evolved)
  - /Users/vivi/qbya-project/qbya/evaluacion/answers.csv
  - /Users/vivi/qbya-project/qbya/evaluacion/questions.csv
  - Chunks: /Users/vivi/qbya-project/qbya/out_chunks/chunks.jsonl

Salidas:
  - /Users/vivi/qbya-project/qbya/evaluacion/grounded_eval.jsonl
  - /Users/vivi/qbya-project/qbya/evaluacion/grounded_eval.csv
  - /Users/vivi/qbya-project/qbya/evaluacion/filtered_questions.csv  (score >= threshold)

Requisitos:
  (.qbya) pip install -U pandas requests
  ollama pull llama3.1:latest
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests

# --- Paths por defecto
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
OUT_CHUNKS = BASE_DIR / "out_chunks" / "chunks.jsonl"
EVAL_DIR = BASE_DIR / "evaluacion"
EVOLVED_CSV = EVAL_DIR / "4-evolved_questions.csv"
ANSWERS_CSV = EVAL_DIR / "2-answers.csv"
QUESTIONS_CSV = EVAL_DIR / "1-questions.csv"
EVAL_JSONL = EVAL_DIR / "5-grounded_eval.jsonl"
EVAL_CSV = EVAL_DIR / "5-grounded_eval.csv"
FILTERED_CSV = EVAL_DIR / "5-filtered_questions.csv"

# --- Prompt (llaves escapadas para str.format)
GROUNDED_PROMPT = """<instrucciones>
<rol>Sos un experto en lingüística y redacción de evaluaciones para modelos de lenguaje.</rol>

<tarea>
Vas a recibir un fragmento de texto (contexto) y una pregunta supuestamente relacionada con ese texto. 
Tu tarea es evaluar en qué medida la pregunta puede responderse utilizando **solo la información disponible en el contexto**.
La evaluación debe ser estricta y seguir las reglas de abajo en el orden dado.
</tarea>

<reglas>
REGLA 1 (PUERTA DURA) — MENCIÓN DE MEDICAMENTO (OBLIGATORIA):
1= Si la pregunta **NO** menciona de forma explícita un medicamento (marca comercial p.ej. "Alercas", o sustancia/monodroga p.ej. "fexofenadina"), entonces el puntaje debe ser **1** automáticamente, sin considerar el contexto.

Definiciones:
- "Explícita" significa que aparece el nombre propio o la sustancia en la redacción de la pregunta.
- No cuentan términos genéricos como "el medicamento", "este fármaco", "antibiótico", "antihistamínico" si no se nombra la droga/marca concreta.

Luego de aplicar la Regla 1, si la pregunta **sí** nombra un medicamento, evaluá la posibilidad de respuesta con **solo** el contexto:
1 = Si la pregunta **NO** menciona de forma explícita un medicamento (marca o droga)
5 = Totalmente y sin ambigüedades respondible solo con el contexto.

Devolvé **únicamente** un JSON con esta estructura (claves en INGLÉS):
{{"explanation": "<una sola oración breve>", "score": <1..5>}}
Sin texto adicional, sin Markdown ni claves extras.
</reglas>

<contexto>
{context}
</contexto>

<pregunta>
{question}
</pregunta>
</instrucciones>""".strip()




# --- Utils de chunks
def read_chunks(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    """chunk_map: id -> obj ; doc_map: doc_key -> [(chunk_id, text), ...]"""
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

def doc_window_around(chunk_id: str, pairs: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
    if k <= 0 or not pairs:
        return pairs
    try:
        base_idx = next(i for i, (cid, _) in enumerate(pairs) if cid == chunk_id)
    except StopIteration:
        base_idx = 0
    left = (k - 1) // 2
    right = k - 1 - left
    start = max(0, base_idx - left)
    end = min(len(pairs), base_idx + right + 1)
    return pairs[start:end]

def ellipsis(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else (text[:max_chars] + "…")

# --- Cliente Ollama
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
        "num_ctx": 16384,     # 🔹 contexto largo fijo (16k tokens)
        "num_predict": 256,   # 🔹 salida acotada
        },
    }
    r = requests.post(url, json=payload, timeout=(timeout_connect, timeout_read))
    if r.status_code == 404:
        avail = ollama_list_models(base_url)
        raise RuntimeError(
            f"Modelo de Ollama '{model}' no disponible (404). "
            f"Instalados: {', '.join(avail) if avail else '(ninguno)'}"
        )
    r.raise_for_status()
    return r.json()

def parse_grounded_from_resp(resp: dict) -> Tuple[str, int]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    content = content.strip()

    def _extract(d: dict) -> Tuple[str, int]:
        # Acepta claves en EN ("explanation","score") y ES ("explicación","puntaje")
        exp = str(d.get("explanation") or d.get("explicación") or "").strip()
        sc  = d.get("score")
        if sc is None:
            sc = d.get("puntaje")
        try:
            sc = int(sc)
        except Exception:
            sc = 3
        return exp, sc

    # 1) JSON directo
    try:
        data = json.loads(content)
        return _extract(data)
    except Exception:
        pass

    # 2) primer bloque { ... }
    start = content.find("{"); end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            return _extract(data)
        except Exception:
            pass

    # 3) fallback
    return content.replace("\n", " ")[:300], 3


def clamp_score(sc: int) -> int:
    try:
        sc = int(sc)
    except Exception:
        return 3
    return max(1, min(5, sc))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(OUT_CHUNKS))
    ap.add_argument("--evolved", default=str(EVOLVED_CSV), help="CSV con question_evolved (si existe).")
    ap.add_argument("--answers", default=str(ANSWERS_CSV))
    ap.add_argument("--questions", default=str(QUESTIONS_CSV))
    ap.add_argument("--out-jsonl", default=str(EVAL_JSONL))
    ap.add_argument("--out-csv", default=str(EVAL_CSV))
    ap.add_argument("--out-filtered", default=str(FILTERED_CSV))
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--scope", choices=["chunk", "document"], default="chunk")
    ap.add_argument("--doc-window", type=int, default=0, help="Si >0 y scope=document, ventana K centrada en el chunk base.")
    ap.add_argument("--max-context-chars", type=int, default=12000)
    ap.add_argument("--threshold", type=int, default=4, help="Score mínimo para conservar (1..5).")
    ap.add_argument("--print-decisions", action="store_true", help="Imprime question + score + keep/drop.")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    # Asegurar carpetas
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_filtered).parent.mkdir(parents=True, exist_ok=True)

    # Cargar preguntas (preferencia)
    df = None
    if args.evolved and Path(args.evolved).exists():
        df = pd.read_csv(args.evolved)
        # columnas esperadas: chunk_id, question_evolved
        if "chunk_id" not in df.columns or ("question_evolved" not in df.columns and "question" not in df.columns):
            raise RuntimeError(f"{args.evolved} debe tener 'chunk_id' y 'question_evolved' o 'question'.")
        use_col = "question_evolved" if "question_evolved" in df.columns else "question"
    elif Path(args.answers).exists():
        df = pd.read_csv(args.answers)
        if "chunk_id" not in df.columns or "question" not in df.columns:
            raise RuntimeError(f"{args.answers} debe tener 'chunk_id' y 'question'.")
        use_col = "question"
    else:
        df = pd.read_csv(args.questions)
        if "chunk_id" not in df.columns or "question" not in df.columns:
            raise RuntimeError(f"{args.questions} debe tener 'chunk_id' y 'question'.")
        use_col = "question"

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # Cargar chunks
    chunk_map, doc_map = read_chunks(Path(args.chunks))
    if not chunk_map:
        raise RuntimeError(f"No se cargaron chunks desde {args.chunks}")

    # Modelos
    models = ollama_list_models(args.ollama_base_url)
    if args.model not in models and f"{args.model}:latest" in models:
        print(f"[INFO] Usando '{args.model}:latest' en lugar de '{args.model}'", flush=True)
        args.model = f"{args.model}:latest"
    elif args.model not in models:
        sys.stderr.write(f"[WARN] Modelo '{args.model}' no está en ollama list. Intento igual…\n")

    rows_eval = []
    kept_rows = []
    t0 = time.time()
    total = len(df)

    for i, row in df.iterrows():
        chunk_id = str(row.get("chunk_id") or "").strip()
        q = str(row.get(use_col) or "").strip()
        if not chunk_id or not q:
            sys.stderr.write(f"[WARN] Fila {i}: falta chunk_id o pregunta. Salto.\n")
            continue

        ch = chunk_map.get(chunk_id)
        if not ch:
            sys.stderr.write(f"[WARN] Fila {i}: chunk_id={chunk_id} no está en chunks.jsonl. Salto.\n")
            continue

        if args.scope == "chunk":
            context = get_text_from_chunk(ch)
        else:
            md = ch.get("metadata") or {}
            doc_key = md.get("doc_id") or md.get("source") or md.get("file") or f"__chunk__:{chunk_id}"
            pairs = doc_map.get(doc_key, []) or [(chunk_id, get_text_from_chunk(ch))]
            if args.doc_window > 0:
                pairs = doc_window_around(chunk_id, pairs, args.doc_window)
            context = "\n\n".join(txt for _, txt in pairs)

        if not context.strip():
            sys.stderr.write(f"[WARN] Fila {i}: contexto vacío. Salto.\n")
            continue

        context_capped = ellipsis(context, args.max_context_chars)
        user_prompt = GROUNDED_PROMPT.format(context=context_capped, question=q)
        system_prompt = "You must output strict JSON with keys 'explanation' and 'score'."

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
            explanation, score = parse_grounded_from_resp(resp)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}): {e}\n")
            explanation, score = "Model error.", 3

        score = clamp_score(score)

        keep = score >= max(1, min(5, int(args.threshold)))
        if args.print_decisions:
            print(f"[{i+1}/{total}] score={score} keep={keep}  Q: {q}", flush=True)

        rows_eval.append({
            "chunk_id": chunk_id,
            "question": q,
            "score": score,
            "explanation": explanation,
        })

        if keep:
            # Conservamos la pregunta en el CSV filtrado
            kept_rows.append({
                "chunk_id": chunk_id,
                "question": q,
            })

        if args.log_every and ((i + 1) % args.log_every == 0):
            elapsed = time.time() - t0
            print(f"[{i+1}/{total}] evaluadas - t={elapsed:.1f}s", flush=True)

    # Guardar evaluaciones
    with Path(args.out_jsonl).open("w", encoding="utf-8") as jf:
        for r in rows_eval:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(rows_eval).to_csv(Path(args.out_csv), index=False, encoding="utf-8")

    # Guardar filtradas
    pd.DataFrame(kept_rows).to_csv(Path(args.out_filtered), index=False, encoding="utf-8")

    print(f"✓ Evaluaciones -> {args.out_jsonl}")
    print(f"✓ CSV eval -> {args.out_csv}")
    print(f"✓ Filtradas (score >= {args.threshold}) -> {args.out_filtered}")
    print(f"Totales: {len(rows_eval)} | Conservadas: {len(kept_rows)} | Descartadas: {len(rows_eval)-len(kept_rows)}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
