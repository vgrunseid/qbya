#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6-evaluar_correctitud.py — Evalúa correctitud del RAG y métricas extra:

Fuentes de preguntas (elegir con --input-source):
  - grounded_csv      -> 5-grounded_eval.csv (filtrando por score >= threshold)
  - grounded_jsonl    -> 5-grounded_eval.jsonl (keep==true o score>=threshold)
  - questions_jsonl   -> 1-questions.jsonl (sin filtro de groundedness)

Ground truth:
  - 2-answers.csv (columna 'answer'), join por chunk_id

Respuesta del RAG:
  - ejecutando 4-retrieve.py -q "<pregunta>" -p ... -c ... --answer --output {answer|full}

Métricas:
  - Correctness (ground truth vs respuesta)  [siempre]
  - Relevance (pregunta vs respuesta)        [--eval-relevance]
  - Groundedness (FACTS vs respuesta)        [--eval-groundedness]   <- FACTS = answer de 3-relevant_fragments.jsonl
  - Retrieval relevance (pregunta vs FACTS)  [--eval-retrieval]      <- FACTS = answer de 3-relevant_fragments.jsonl

Notas:
  - Para groundedness/retrieval se usa 3-relevant_fragments.jsonl: se toma el campo "answer" del registro con el chunk_id evaluado.
  - Si falta ese chunk_id o "answer" está vacío, FACTS quedará vacío.

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
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import requests


# -------- Paths por defecto (ajustados a tu proyecto)
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
EVAL_DIR = BASE_DIR / "evaluacion"

GROUNDED_EVAL_CSV   = EVAL_DIR / "5-grounded_eval.csv"
GROUNDED_EVAL_JSONL = EVAL_DIR / "5-grounded_eval.jsonl"
QUESTIONS_JSONL     = EVAL_DIR / "1-questions.jsonl"
RELEVANT_FRAGMENTS_JSONL = EVAL_DIR / "3-relevant_fragments.jsonl"

ANSWERS_CSV  = EVAL_DIR / "2-answers.csv"   # ground truth en columna 'answer'
RETRIEVER_PY = BASE_DIR / "4-retrieve_mq.py"

OUT_JSONL = EVAL_DIR / "6-correctness_eval.jsonl"
OUT_CSV   = EVAL_DIR / "6-correctness_eval.csv"


# -------- Prompts (JSON estricto; llaves escapadas donde corresponde)
CORRECTNESS_PROMPT = """<instrucciones>
<rol>Sos un docente que está calificando un examen.</rol>

<tarea>
Vas a recibir una PREGUNTA, la RESPUESTA CORRECTA (ground truth) y la RESPUESTA DEL ESTUDIANTE.
</tarea>

<reglas>
Calificá ÚNICAMENTE la precisión factual de la respuesta del estudiante en relación con la respuesta correcta.
No penalices si el estudiante agrega información adicional, siempre que no contradiga la respuesta correcta.

Reglas de decisión de correctitud (aplicalas en orden):

1) Si la RESPUESTA CORRECTA contiene un valor puntual (por ejemplo, cantidades numéricas con o sin unidad), entonces:
   - La RESPUESTA DEL ESTUDIANTE solo es correcta si incluye explícitamente ese mismo valor (permitiendo variantes como coma/punto/espacio/agrupadores).
   - Ejemplo: "31,000 mg" == "31000 mg" == "31.000 mg" == "31 000 mg".
   - Si la respuesta del estudiante no incluye ese valor o afirma que “no hay suficiente información”, marcala como incorrecta.

2) Si la RESPUESTA CORRECTA es textual no numérica:
   - Marcala como correcta si la RESPUESTA DEL ESTUDIANTE expresa exactamente el mismo significado mediante una redacción equivalente.
   - Si falta la información esencial o se contradice, marcala como incorrecta.

3) Si la RESPUESTA DEL ESTUDIANTE enumera varios valores y NO incluye el valor del ground truth, marcala como incorrecta.

Devolvé ÚNICAMENTE un JSON con esta estructura:
{{"explanation": "<una sola oración breve>", "is_correct": <true|false>}}
Sin formato Markdown, sin texto adicional ni claves extra.
</reglas>

PREGUNTA:
{question}

RESPUESTA CORRECTA:
{ground_truth_answer}

RESPUESTA DEL ESTUDIANTE:
{answer}
</tarea>
</instrucciones>""".strip()


RELEVANCE_PROMPT = """<instrucciones>
<rol>Sos un docente que está calificando un examen.</rol>

<tarea>
Vas a recibir una PREGUNTA y una RESPUESTA DEL ESTUDIANTE.
</tarea>

<reglas>
Evaluá si la RESPUESTA DEL ESTUDIANTE responde directamente a la PREGUNTA y si es concisa y relevante.
- Si la respuesta ayuda a responder la pregunta y se mantiene en el tema, es relevante.
- Si no responde la pregunta o se desvía del tema, no es relevante.

Devolvé ÚNICAMENTE un JSON con esta estructura:
{{"explanation": "<una sola oración breve>", "is_relevant": <true|false>}}
Sin formato Markdown, sin texto adicional ni claves extra.
</reglas>

PREGUNTA:
{question}

RESPUESTA DEL ESTUDIANTE:
{answer}
</tarea>
</instrucciones>""".strip()


GROUNDEDNESS_PROMPT = """<instrucciones>
<rol>Sos un docente que está calificando un examen.</rol>

<tarea>
Vas a recibir un conjunto de HECHOS (FACTS) y una RESPUESTA DEL ESTUDIANTE.
</tarea>

<reglas>
Determiná si la RESPUESTA DEL ESTUDIANTE está completamente fundamentada en los HECHOS y no introduce información que no esté respaldada.
- Si todas las afirmaciones de la respuesta están sustentadas por los hechos, marcala como fundamentada (grounded).
- Si la respuesta contiene información que no aparece en los hechos, marcala como no fundamentada (not grounded).

Devolvé ÚNICAMENTE un JSON con esta estructura:
{{"explanation": "<una sola oración breve>", "is_grounded": <true|false>}}
Sin formato Markdown, sin texto adicional ni claves extra.
</reglas>

HECHOS:
{doc_string}

RESPUESTA DEL ESTUDIANTE:
{answer}
</tarea>
</instrucciones>""".strip()


RETRIEVAL_RELEVANCE_PROMPT = """<instrucciones>
<rol>Sos un docente que está calificando un examen.</rol>

<tarea>
Vas a recibir un conjunto de HECHOS (fragmentos recuperados) y una PREGUNTA.
</tarea>

<reglas>
Evaluá si los HECHOS son relevantes para la PREGUNTA.
- Si los hechos contienen palabras clave o información semánticamente relacionada con la pregunta, se consideran relevantes (aunque sea parcialmente).
- Si los hechos no tienen ninguna relación con la pregunta, se consideran no relevantes.

Devolvé ÚNICAMENTE un JSON con esta estructura:
{{"explanation": "<una sola oración breve>", "is_retrieval_relevant": <true|false>}}
Sin formato Markdown, sin texto adicional ni claves extra.
</reglas>

HECHOS:
{doc_string}

PREGUNTA:
{question}
</tarea>
</instrucciones>""".strip()



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


# -------- Parsers JSON (tolerantes a respuestas no estrictas)
def _parse_bool_field(content: str, key: str) -> Tuple[str, Optional[bool]]:
    content = (content or "").strip()
    # 1) JSON directo
    try:
        data = json.loads(content)
        return str(data.get("explanation", "")).strip(), (bool(data.get(key)) if key in data else None)
    except Exception:
        pass
    # 2) buscar bloque { ... }
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            return str(data.get("explanation", "")).strip(), (bool(data.get(key)) if key in data else None)
        except Exception:
            pass
    # 3) fallback heurístico muy básico
    lc = content.lower()
    guess = True if "true" in lc and "false" not in lc else (False if "false" in lc and "true" not in lc else None)
    return content.replace("\n", " ")[:300], guess

def parse_correctness_from_resp(resp: dict) -> Tuple[str, bool]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    expl, val = _parse_bool_field(content, "is_correct")
    return expl, bool(val)

def parse_relevance_from_resp(resp: dict) -> Tuple[str, Optional[bool]]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    return _parse_bool_field(content, "is_relevant")

def parse_groundedness_from_resp(resp: dict) -> Tuple[str, Optional[bool]]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    return _parse_bool_field(content, "is_grounded")

def parse_retrieval_rel_from_resp(resp: dict) -> Tuple[str, Optional[bool]]:
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    return _parse_bool_field(content, "is_retrieval_relevant")


# -------- Helpers: carga de relevant fragments (FACTS)
def load_relevant_facts_map(path: Path, max_chars: int) -> Dict[str, str]:
    """
    Carga 3-relevant_fragments.jsonl y devuelve: chunk_id -> doc_string (FACTS)
      - Usa preferentemente el campo 'answer' como contexto.
      - Si 'answer' no existe, intenta 'sentences' (lista) o 'sentences_join'.
    """
    mp: Dict[str, str] = {}
    if not path.exists():
        return mp
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = str(obj.get("chunk_id") or "").strip()
            if not cid:
                continue
            ctx = obj.get("answer")
            if not ctx:
                sents = obj.get("sentences") or []
                if isinstance(sents, list) and sents:
                    ctx = " ".join([str(s).strip() for s in sents if isinstance(s, str)])
                elif obj.get("sentences_join"):
                    ctx = str(obj.get("sentences_join"))
            if not ctx:
                continue
            ctx = str(ctx).strip()
            if len(ctx) > max_chars:
                ctx = ctx[:max_chars] + "…"
            mp[cid] = ctx
    return mp


# -------- RAG runner (4-retrieve.py)
def run_retriever(
    retriever_py: Path,
    question: str,
    path: str,
    collection: str,
    extra_args: Optional[str],
    timeout: float = 120.0,
    cwd: Optional[Path] = None,
    echo_cmd: bool = False,
    rag_output: str = "answer",  # "answer" | "full"
) -> Tuple[str, str]:
    """
    Ejecuta 4-retrieve.py con --output {answer|full}. Siempre pasa --answer.
    Devuelve (answer, stdout_completo).
    """
    cmd_parts = [
        "python", str(Path(retriever_py)),
        "-q", question,
        "-p", path,
        "-c", collection,
        "--answer",
        "--output", rag_output,
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

    answer = stdout.strip()
    return answer, stdout


# -------- Utilidad para recortar strings
def ellipsis(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    t = str(text)
    return t if len(t) <= max_chars else (t[:max_chars] + "…")


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--grounded", default=str(GROUNDED_EVAL_CSV),
                    help="CSV con preguntas y score (5-grounded_eval.csv).")
    ap.add_argument("--grounded-jsonl", default=str(GROUNDED_EVAL_JSONL),
                    help="JSONL con evaluaciones (5-grounded_eval.jsonl).")
    ap.add_argument("--questions-jsonl", default=str(QUESTIONS_JSONL),
                    help="JSONL con preguntas crudas (1-questions.jsonl).")
    ap.add_argument("--relevant-fragments", default=str(RELEVANT_FRAGMENTS_JSONL),
                    help="JSONL con fragmentos relevantes (3-relevant_fragments.jsonl).")
    ap.add_argument("--answers", default=str(ANSWERS_CSV),
                    help="CSV con ground truth en columna 'answer'.")
    ap.add_argument("--threshold", type=int, default=4,
                    help="Filtrar preguntas con score >= threshold (default 4).")

    ap.add_argument("--input-source",
                    choices=["grounded_csv", "grounded_jsonl", "questions_jsonl"],
                    default="grounded_csv",
                    help="Fuente de preguntas a evaluar.")

    # retriever
    ap.add_argument("--retriever", default=str(RETRIEVER_PY), help="Ruta a 4-retrieve.py")
    ap.add_argument("--retriever-cwd", default=str(BASE_DIR),
                    help="Directorio de trabajo para ejecutar 4-retrieve.py (default: raíz del proyecto).")
    ap.add_argument("--path", default="chroma/prospectos", help="-p para 4-retrieve.py")
    ap.add_argument("--collection", default="prospectos", help="-c para 4-retrieve.py")
    ap.add_argument("--retriever-extra-args", default="--debug --rerank cross", help="Args extra a pasarle a 4-retrieve.py")
    ap.add_argument("--retriever-timeout", type=float, default=120.0)
    ap.add_argument("--rag-output", choices=["answer", "full"], default="answer",
                    help="Salida de 4-retrieve.py (no necesaria para FACTS).")

    ap.add_argument("--echo-cmd", action="store_true",
                    help="Imprime el comando exacto y el CWD al ejecutar 4-retrieve.py.")
    ap.add_argument("--include-rag-stdout", action="store_true",
                    help="Agrega la salida completa de 4-retrieve.py en el CSV.")

    # ollama
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.0)

    # control
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)

    # facts (desde 3-relevant_fragments.jsonl)
    ap.add_argument("--facts-max-chars", type=int, default=12000,
                    help="Tope de caracteres para el FACTS doc_string.")

    # eval toggles
    ap.add_argument("--eval-relevance", action="store_true",
                    help="Evalúa Relevance (question vs answer).")
    ap.add_argument("--eval-groundedness", action="store_true",
                    help="Evalúa Groundedness (facts vs answer).")
    ap.add_argument("--eval-retrieval", action="store_true",
                    help="Evalúa Retrieval relevance (facts vs question).")

    # out
    ap.add_argument("--out-jsonl", default=str(OUT_JSONL))
    ap.add_argument("--out-csv", default=str(OUT_CSV))

    ap.add_argument("--print-facts-flag", action="store_true",
                help="Imprime por cada fila si se recuperó el relevant_fragment (answer) para ese chunk_id.")

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

    # --- Cargar relevant fragments (FACTS) por chunk_id ---
    rf_path = Path(args.relevant_fragments)
    facts_map = load_relevant_facts_map(rf_path, max_chars=args.facts_max_chars)
    if (args.eval_groundedness or args.eval_retrieval) and not facts_map:
        print(f"[WARN] No se pudo cargar FACTS desde {rf_path}; groundedness/retrieval usarán FACTS vacíos.", flush=True)

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
    rows_out: List[Dict[str, Any]] = []
    correct = 0
    total = len(df)
    t0 = time.time()

    for i, r in df.iterrows():
        chunk_id = str(r["chunk_id"]).strip()
        question = str(r["question"]).strip()
        gt = str(r["ground_truth_answer"]).strip()

        # 1) Ejecutar RAG para obtener respuesta del sistema
        try:
            rag_answer, rag_stdout = run_retriever(
                retriever_py=Path(args.retriever),
                question=question,
                path=args.path,
                collection=args.collection,
                extra_args=args.retriever_extra_args,
                timeout=args.retriever_timeout,
                cwd=Path(args.retriever_cwd),
                echo_cmd=args.echo_cmd,
                rag_output=args.rag_output,
            )
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}) al correr 4-retrieve.py: {e}\n")
            rag_answer, rag_stdout = "", ""
        print(rag_answer)
        # 2) Correctness
        user_prompt = CORRECTNESS_PROMPT.format(
            question=question, ground_truth_answer=gt, answer=rag_answer
        )
        system_prompt = "Return strict JSON with keys 'explanation' and 'is_correct'. No extra text."
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
            corr_expl, is_correct = parse_correctness_from_resp(resp)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Fila {i} (chunk_id={chunk_id}) evaluando correctitud: {e}\n")
            corr_expl, is_correct = "Model error.", False

        print(f"\n[{i+1}/{total}] Pregunta: {question}")
        print(f"Respuesta correcta (ground truth): {gt}")
        print(" ")
        print("Respuesta del RAG:")
        print(rag_answer if rag_answer.strip() else "(sin respuesta)")
        status = "✅ CORRECTA" if is_correct else "❌ INCORRECTA"
        print(f"[{i+1}/{total}] {status}  chunk={chunk_id}")


        if is_correct:
            correct += 1

        row: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "question": question,
            "ground_truth_answer": gt,
            "rag_answer": rag_answer,
            "is_correct": bool(is_correct),
            "correctness_explanation": corr_expl,
        }

        # 3) Relevance (opcional)
        if args.eval_relevance:
            rel_prompt = RELEVANCE_PROMPT.format(question=question, answer=rag_answer)
            try:
                resp = ollama_chat_json(
                    base_url=args.ollama_base_url,
                    model=args.model,
                    system_prompt="Return strict JSON with keys 'explanation' and 'is_relevant'.",
                    user_prompt=rel_prompt,
                    temperature=args.temperature,
                    timeout_connect=10.0,
                    timeout_read=120.0,
                )
                rel_expl, is_relevant = parse_relevance_from_resp(resp)
            except Exception as e:
                sys.stderr.write(f"[WARN] Relevance fallo en fila {i}: {e}\n")
                rel_expl, is_relevant = "Model error.", None
            row.update({"is_relevant": is_relevant, "relevance_explanation": rel_expl})

        # 4) Groundedness / Retrieval relevance (opcionales) usando FACTS del JSONL
        if args.eval_groundedness or args.eval_retrieval:
            facts_str = ellipsis(facts_map.get(chunk_id, ""), args.facts_max_chars)

            facts_ok = bool(facts_map.get(chunk_id))
            if args.print_facts_flag:
                print(f"[FACTS] {i+1}/{total} chunk_id={chunk_id} found={facts_ok} chars={len(facts_str)}", flush=True)

                # Guardá la bandera en la salida para CSV/JSONL
                row["facts_found"] = facts_ok

            if args.eval_groundedness:
                grd_prompt = GROUNDEDNESS_PROMPT.format(doc_string=facts_str, answer=rag_answer)
                try:
                    resp = ollama_chat_json(
                        base_url=args.ollama_base_url,
                        model=args.model,
                        system_prompt="Return strict JSON with keys 'explanation' and 'is_grounded'.",
                        user_prompt=grd_prompt,
                        temperature=args.temperature,
                        timeout_connect=10.0,
                        timeout_read=120.0,
                    )
                    grd_expl, is_grounded = parse_groundedness_from_resp(resp)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Groundedness fallo en fila {i}: {e}\n")
                    grd_expl, is_grounded = "Model error.", None
                row.update({"is_grounded": is_grounded, "groundedness_explanation": grd_expl, "facts_excerpt": facts_str})

            if args.eval_retrieval:
                rr_prompt = RETRIEVAL_RELEVANCE_PROMPT.format(doc_string=facts_str, question=question)
                try:
                    resp = ollama_chat_json(
                        base_url=args.ollama_base_url,
                        model=args.model,
                        system_prompt="Return strict JSON with keys 'explanation' and 'is_retrieval_relevant'.",
                        user_prompt=rr_prompt,
                        temperature=args.temperature,
                        timeout_connect=10.0,
                        timeout_read=120.0,
                    )
                    rr_expl, is_ret_rel = parse_retrieval_rel_from_resp(resp)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Retrieval relevance fallo en fila {i}: {e}\n")
                    rr_expl, is_ret_rel = "Model error.", None
                row.update({"is_retrieval_relevant": is_ret_rel, "retrieval_explanation": rr_expl})
                if "facts_excerpt" not in row:
                    row["facts_excerpt"] = facts_str

        if args.include_rag_stdout:
            row["retriever_stdout_raw"] = rag_stdout

        rows_out.append(row)

        if args.log_every and ((i + 1) % args.log_every == 0):
            elapsed = time.time() - t0
            acc = 100.0 * correct / (i + 1)
            print(f"[{i+1}/{total}] acc_prom={acc:.1f}%  t={elapsed:.1f}s", flush=True)



        if args.eval_relevance:
            print(f"Relevancia de la respuesta: {'💬 Relevante' if is_relevant else '🚫 No relevante'} — {rel_expl}")

        if args.eval_groundedness:
            print(f"Fundamentación (groundedness): {'🧩 Correcta' if is_grounded else '⚠️ No fundamentada'} — {grd_expl}")

        if args.eval_retrieval:
            print(f"Relevancia de recuperación: {'🔍 Relevante' if is_ret_rel else '🚫 No relevante'} — {rr_expl}")

        print("--------------------------------------------------------------")


    # --- Guardar resultados
    with Path(args.out_jsonl).open("w", encoding="utf-8") as jf:
        for r in rows_out:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(rows_out).to_csv(Path(args.out_csv), index=False, encoding="utf-8")

    # --- Resumen en consola: imprime hasta 4 accuracies según flags
    def _metric(rows: List[Dict[str, Any]], key: str) -> Tuple[int, int, float]:
        vals = [r.get(key) for r in rows if key in r]
        vals = [v for v in vals if isinstance(v, bool)]
        denom = len(vals)
        truec = sum(v for v in vals)  # True=1, False=0
        pct = (100.0 * truec / denom) if denom else 0.0
        return truec, denom, pct

    def _print_metric(name: str, key: str):
        truec, denom, pct = _metric(rows_out, key)
        if denom:
            print(f"{name}: {pct:.2f}%  ({truec}/{denom})")
        else:
            print(f"{name}: n/a (0 válidos)")

    print(f"✓ Correctitud -> {args.out_jsonl}")
    print(f"✓ CSV -> {args.out_csv}")

    # Siempre correctness
    _print_metric("Accuracy (Correctness)", "is_correct")
    if args.eval_relevance:
        _print_metric("Accuracy (Relevance)", "is_relevant")
    if args.eval_groundedness:
        _print_metric("Accuracy (Groundedness)", "is_grounded")
    if args.eval_retrieval:
        _print_metric("Accuracy (Retrieval relevance)", "is_retrieval_relevant")

    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
