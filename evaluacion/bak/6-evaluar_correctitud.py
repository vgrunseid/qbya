#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6-evaluar_correctitud.py — Evalúa correctitud del RAG usando:
- Fuente de preguntas configurable:
    * grounded_csv      -> 5-grounded_eval.csv (filtrando por score >= threshold)
    * grounded_jsonl    -> 5-grounded_eval.jsonl (keep==true o score>=threshold)
    * questions_jsonl   -> 1-questions.jsonl (sin filtro de groundedness)
- Ground truth: answers.csv (columna 'answer'), join por chunk_id
- Respuesta del RAG: ejecutando 4-retrieve.py -q "<pregunta>" -p ... -c ... --answer --output answer

Entradas por defecto:
  - /Users/vivi/qbya-project/qbya/evaluacion/5-grounded_eval.csv
  - /Users/vivi/qbya-project/qbya/evaluacion/5-grounded_eval.jsonl
  - /Users/vivi/qbya-project/qbya/evaluacion/1-questions.jsonl
  - /Users/vivi/qbya-project/qbya/evaluacion/2-answers.csv
  - /Users/vivi/qbya-project/qbya/4-retrieve.py

Salidas:
  - /Users/vivi/qbya-project/qbya/evaluacion/6-correctness_eval.jsonl
  - /Users/vivi/qbya-project/qbya/evaluacion/6-correctness_eval.csv
  - imprime Accuracy global

Requisitos:
  (.qbya) pip install -U pandas requests
  ollama pull llama3.1:latest
"""

import argparse
import json
import sys
import time
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests

# -------- Paths por defecto (ajustados a tu proyecto)
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
EVAL_DIR = BASE_DIR / "evaluacion"

GROUNDED_EVAL_CSV   = EVAL_DIR / "5-grounded_eval.csv"
GROUNDED_EVAL_JSONL = EVAL_DIR / "5-grounded_eval.jsonl"
QUESTIONS_JSONL     = EVAL_DIR / "1-questions.jsonl"

ANSWERS_CSV = EVAL_DIR / "2-answers.csv"   # ground truth en columna 'answer'
RETRIEVER_PY = BASE_DIR / "4-retrieve.py"

OUT_JSONL = EVAL_DIR / "6-correctness_eval.jsonl"
OUT_CSV   = EVAL_DIR / "6-correctness_eval.csv"

# -------- Prompt (devuelve JSON estricto)
CORRECTNESS_PROMPT = """<instructions>
<role>You are a teacher grading a quiz.</role>
<task>
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

<rules>
Grade ONLY factual accuracy of the student answer relative to the ground truth.
Do NOT penalize the student for adding extra information if it does not contradict the ground truth.

Correctness decision rules (apply in order):
1) If the STUDENT ANSWER explicitly contains the GROUND TRUTH value anywhere in the text, mark it as correct.
   - Consider minor formatting/locale variations as equivalent (e.g., decimal comma vs dot, thin/thousands separators, extra spaces, case/diacritics).
   - Example: "1,279 mg" == "1.279 mg" == "1 279 mg".
2) If the STUDENT ANSWER provides an equivalent wording/paraphrase of the GROUND TRUTH (same meaning), mark it as correct.
3) Only if the requested specific information is missing or contradicted, mark it as incorrect.

Output JSON ONLY: {{\"explanation\":\"<one concise sentence>\", \"is_correct\": <true|false>}}.
No markdown, no extra keys, no additional text.
</rules>

QUESTION:
{question}

GROUND TRUTH:
{ground_truth_answer}

STUDENT ANSWER:
{answer}
</task>
</instructions>""".strip()

# -------- Ollama (HTTP)
def ollama_list_models(base_url: str, timeout: float = 10.0) -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m.get("model") or m.get("name") for m in data.get("models", []) if m.get("model") or m.get("name")]
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
        "options": {"temperature": float(temperature)},
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

def parse_correctness_from_resp(resp: dict) -> Tuple[str, bool]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    content = content.strip()

    # JSON directo
    try:
        data = json.loads(content)
        expl = str(data.get("explanation", "")).strip()
        is_corr = bool(data.get("is_correct"))
        return expl, is_corr
    except Exception:
        pass

    # primer bloque { ... }
    start = content.find("{"); end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            expl = str(data.get("explanation", "")).strip()
            is_corr = bool(data.get("is_correct"))
            return expl, is_corr
        except Exception:
            pass

    # fallback heurístico mínimo
    lc = content.lower()
    is_corr = (" true" in lc) or ("\"is_correct\": true" in lc) or ("correct\": true" in lc)
    return content.replace("\n", " ")[:300], is_corr

# -------- RAG runner (4-retrieve.py) — con CWD y logs
def run_retriever_and_get_answer(
    retriever_py: Path,
    question: str,
    path: str,
    collection: str,
    extra_args: Optional[str],
    timeout: float = 120.0,
    cwd: Optional[Path] = None,
    echo_cmd: bool = False,
) -> Tuple[str, str]:
    """
    Ejecuta:
      python 4-retrieve.py -q "<question>" -p <path> -c <collection> --answer --output answer [extra_args]
    en el directorio 'cwd' (recomendado: raíz del proyecto).
    Devuelve (respuesta_parseada, stdout_completo).
    """
    cmd_parts = [
        "python", str(Path(retriever_py)),
        "-q", question,
        "-p", path,
        "-c", collection,
        "--answer",
        "--output", "answer",
    ]
    if extra_args:
        cmd_parts += shlex.split(extra_args)

    cmd_str = " ".join(shlex.quote(x) for x in cmd_parts)
    run_cwd = str(cwd) if cwd else None
    if echo_cmd:
        print(f"[CMD] {cmd_str}\n[CWD] {run_cwd or '(current)'}", flush=True)

    try:
        proc = subprocess.run(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=run_cwd,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"4-retrieve.py timeout.\n[CMD] {cmd_str}\n[CWD] {run_cwd}")

    stdout = (proc.stdout or "")
    stderr = (proc.stderr or "")

    if proc.returncode != 0:
        snip_len = 2000
        raise RuntimeError(
            "4-retrieve.py error (code {}).\n[CMD] {}\n[CWD] {}\n--- STDERR ---\n{}\n--- STDOUT ---\n{}".format(
                proc.returncode, cmd_str, run_cwd,
                stderr.strip()[:snip_len],
                stdout.strip()[:snip_len],
            )
        )

    # Como --output answer imprime SOLO la respuesta, usamos stdout entero:
    answer = stdout.strip()
    return answer, stdout

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--grounded", default=str(GROUNDED_EVAL_CSV),
                    help="CSV con preguntas y score (5-grounded_eval.csv).")
    ap.add_argument("--grounded-jsonl", default=str(GROUNDED_EVAL_JSONL),
                    help="JSONL con evaluaciones (5-grounded_eval.jsonl).")
    ap.add_argument("--questions-jsonl", default=str(QUESTIONS_JSONL),
                    help="JSONL con preguntas crudas (1-questions.jsonl).")
    ap.add_argument("--answers", default=str(ANSWERS_CSV),
                    help="CSV con ground truth en columna 'answer'.")
    ap.add_argument("--threshold", type=int, default=4,
                    help="Filtrar preguntas con score >= threshold (default 4).")

    ap.add_argument("--input-source",
                    choices=["grounded_csv", "grounded_jsonl", "questions_jsonl"],
                    default="grounded_csv",
                    help="Fuente de preguntas a evaluar.")

    # retriever (ruta fija y CWD a tu proyecto)
    ap.add_argument("--retriever", default=str(RETRIEVER_PY), help="Ruta a 4-retrieve.py")
    ap.add_argument("--retriever-cwd", default=str(BASE_DIR),
                    help="Directorio de trabajo para ejecutar 4-retrieve.py (default: raíz del proyecto).")
    ap.add_argument("--path", default="chroma/prospectos", help="-p para 4-retrieve.py")
    ap.add_argument("--collection", default="prospectos", help="-c para 4-retrieve.py")
    ap.add_argument("--retriever-extra-args", default="--debug", help="Args extra a pasarle a 4-retrieve.py")
    ap.add_argument("--retriever-timeout", type=float, default=120.0)
    ap.add_argument("--echo-cmd", action="store_true",
                    help="Imprime el comando exacto y el CWD al ejecutar 4-retrieve.py.")
    ap.add_argument("--include-rag-stdout", action="store_true",
                    help="Agrega la salida completa de 4-retrieve.py en el CSV (columna retriever_stdout_raw).")
    # ollama
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.0)
    # control
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    # out
    ap.add_argument("--out-jsonl", default=str(OUT_JSONL))
    ap.add_argument("--out-csv", default=str(OUT_CSV))

    args = ap.parse_args()

    # Validaciones mínimas de rutas comunes
    answers_path  = Path(args.answers)
    retriever_py  = Path(args.retriever)
    retriever_cwd = Path(args.retriever_cwd)

    if not answers_path.exists():
        raise RuntimeError(f"No existe answers.csv en {answers_path}")
    if not retriever_py.exists():
        raise RuntimeError(f"No encuentro 4-retrieve.py en {retriever_py}")
    if not retriever_cwd.exists():
        raise RuntimeError(f"El CWD para ejecutar 4-retrieve.py no existe: {retriever_cwd}")

    # --- Cargar fuente de PREGUNTAS según --input-source ---
    src = args.input_source
    if src == "grounded_csv":
        grounded_path = Path(args.grounded)
        if not grounded_path.exists():
            raise RuntimeError(f"No existe grounded_eval.csv en {grounded_path}")
        df_g = pd.read_csv(grounded_path)
        req_g = {"chunk_id", "question", "score"}
        if not req_g.issubset(df_g.columns):
            raise RuntimeError(f"{grounded_path} debe tener columnas {req_g}. Tiene: {df_g.columns.tolist()}")
        df_q = df_g[df_g["score"] >= int(args.threshold)][["chunk_id", "question"]].copy()

    elif src == "grounded_jsonl":
        gj_path = Path(args.grounded_jsonl)
        if not gj_path.exists():
            raise RuntimeError(f"No existe grounded_eval.jsonl en {gj_path}")
        rows = []
        with gj_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("chunk_id") or "").strip()
                q   = str(obj.get("question") or "").strip()
                if not cid or not q:
                    continue
                keep = obj.get("keep")
                score = obj.get("score")
                if (isinstance(keep, bool) and keep) or (keep is None and isinstance(score, (int, float)) and int(score) >= int(args.threshold)):
                    rows.append({"chunk_id": cid, "question": q})
        if not rows:
            raise RuntimeError("No hay preguntas seleccionadas desde 5-grounded_eval.jsonl con los criterios dados.")
        df_q = pd.DataFrame(rows)

    else:  # src == "questions_jsonl"
        qj_path = Path(args.questions_jsonl)
        if not qj_path.exists():
            raise RuntimeError(f"No existe questions.jsonl en {qj_path}")
        rows = []
        with qj_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("chunk_id") or (obj.get("metadata") or {}).get("id") or "").strip()
                q   = str(obj.get("question") or "").strip()
                if cid and q:
                    rows.append({"chunk_id": cid, "question": q})
        if not rows:
            raise RuntimeError("No hay preguntas en questions.jsonl.")
        df_q = pd.DataFrame(rows)

    if args.max_rows:
        df_q = df_q.head(args.max_rows).copy()

    # --- Cargar GROUND TRUTH y unir por chunk_id ---
    df_gt = pd.read_csv(answers_path)
    req_gt = {"chunk_id", "answer"}
    if not req_gt.issubset(df_gt.columns):
        raise RuntimeError(f"{answers_path} debe tener columnas {req_gt}. Tiene: {df_gt.columns.tolist()}")
    df_gt = df_gt.rename(columns={"answer": "ground_truth_answer"})

    df = pd.merge(df_q, df_gt[["chunk_id", "ground_truth_answer"]], on="chunk_id", how="inner")
    if df.empty:
        raise RuntimeError("No hay filas tras unir preguntas con answers.csv por chunk_id.")

    # Chequear modelo en Ollama
    models = ollama_list_models(args.ollama_base_url)
    if args.model not in models and f"{args.model}:latest" in models:
        print(f"[INFO] Usando '{args.model}:latest' en lugar de '{args.model}'", flush=True)
        args.model = f"{args.model}:latest"
    elif args.model not in models:
        sys.stderr.write(f"[WARN] Modelo '{args.model}' no aparece en ollama list. Intento igual…\n")

    # Asegurar salidas
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Loop
    rows_out = []
    correct = 0
    total = len(df)
    t0 = time.time()

    for i, r in df.iterrows():
        chunk_id = str(r["chunk_id"]).strip()
        question = str(r["question"]).strip()
        gt = str(r["ground_truth_answer"]).strip()

        # 1) Ejecutar RAG para obtener respuesta del sistema
        try:
            rag_answer, rag_stdout = run_retriever_and_get_answer(
                retriever_py=Path(args.retriever),
                question=question,
                path=args.path,
                collection=args.collection,
                extra_args=args.retriever_extra_args,
                timeout=args.retriever_timeout,
                cwd=Path(args.retriever_cwd),
                echo_cmd=args.echo_cmd,
            )
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}) al correr 4-retrieve.py: {e}\n")
            rag_answer, rag_stdout = "", ""

        # 2) Armar prompt de correctitud
        user_prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth_answer=gt,
            answer=rag_answer
        )
        system_prompt = "Return strict JSON with keys 'explanation' and 'is_correct'. No extra text."

        # 3) Llamar a Ollama para evaluar
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
            explanation, is_correct = parse_correctness_from_resp(resp)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}) evaluando correctitud: {e}\n")
            explanation, is_correct = "Model error.", False

        if is_correct:
            correct += 1

        row = {
            "chunk_id": chunk_id,
            "question": question,
            "ground_truth_answer": gt,
            "rag_answer": rag_answer,
            "is_correct": bool(is_correct),
            "explanation": explanation,
        }
        if args.include_rag_stdout:
            row["retriever_stdout_raw"] = rag_stdout

        rows_out.append(row)

        if args.log_every and ((i + 1) % args.log_every == 0):
            elapsed = time.time() - t0
            acc = 100.0 * correct / (i + 1)
            print(f"[{i+1}/{total}] acc_prom={acc:.1f}%  t={elapsed:.1f}s", flush=True)

    # Guardar resultados
    with Path(args.out_jsonl).open("w", encoding="utf-8") as jf:
        for r in rows_out:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(rows_out).to_csv(Path(args.out_csv), index=False, encoding="utf-8")

    # Resumen
    accuracy = 100.0 * correct / total if total else 0.0
    print(f"✓ Correctitud -> {args.out_jsonl}")
    print(f"✓ CSV -> {args.out_csv}")
    print(f"Accuracy: {accuracy:.2f}%  ({correct}/{total})")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
