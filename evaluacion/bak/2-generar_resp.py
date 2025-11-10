#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generar 1 respuesta por pregunta (questions.csv) usando OLLAMA (llama3.1),
rehidratando el contexto EXACTO desde el chunk original (chunk_id).

Entradas por defecto:
  - Chunks   : /Users/vivi/qbya-project/qbya/out_chunks/chunks.jsonl
  - Preguntas: /Users/vivi/qbya-project/qbya/evaluacion/questions.csv

Salidas por defecto:
  - JSONL: /Users/vivi/qbya-project/qbya/evaluacion/answers.jsonl
  - CSV  : /Users/vivi/qbya-project/qbya/evaluacion/answers.csv

Requisitos:
  (.qbya) pip install -U pandas pydantic langchain-core langchain-ollama
  ollama pull llama3.1:8b  # o el tag que tengas
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# -------------------------
# Paths por defecto
# -------------------------
BASE_DIR = Path("/Users/vivi/qbya-project/qbya")
CHUNKS_JSONL = BASE_DIR / "out_chunks" / "chunks.jsonl"
EVAL_DIR = BASE_DIR / "evaluacion"
QUESTIONS_CSV = EVAL_DIR / "1-questions.csv"
ANSWERS_JSONL = EVAL_DIR / "2-answers.jsonl"
ANSWERS_CSV = EVAL_DIR / "2-answers.csv"


# -------------------------
# Esquema de salida (JSON "fijo")
# -------------------------
class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Final answer grounded only on the provided context")


ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<instructions>
You are a careful, concise medical assistant.
You will answer the question using only the information inside <context>.
If the context does not contain the answer, reply with "Not enough information in the provided context."

<context>
{context}
</context>

<question>
{question}
</question>

<rules>
1) Base your answer strictly on the context (no external knowledge).
2) No links. No citations. No markdown.
3) Keep it concise (1–3 sentences), medically accurate and specific.
4) Reflect dose/posology/naming exactly as in the context when present.
5) If context is insufficient, output exactly: Not enough information in the provided context.
6) Output JSON ONLY with this exact schema (no extra keys, no prose):
   {{"answer": "<your concise answer here>"}}
</rules>
</instructions>"""
)


# -------------------------
# Utilidades de chunks
# -------------------------
def read_chunks_to_map(path: Path) -> Dict[Optional[str], Dict[str, Any]]:
    """
    Construye un mapa: chunk_id -> objeto del chunk.
    Soporta formatos con 'page_content' o 'text'.
    """
    mp: Dict[Optional[str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id") or (obj.get("metadata") or {}).get("id")
            if cid:
                mp[cid] = obj
    return mp


def get_context_from_chunk(obj: Dict[str, Any]) -> str:
    return (obj.get("page_content") or obj.get("text") or "").strip()


# -------------------------
# Cliente LLM (Ollama)
# -------------------------
def make_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(
        model=model,           # p.ej. "llama3.1:8b"
        base_url=base_url,     # default http://localhost:11434
        temperature=temperature,
        num_ctx=8192,
    )


def call_ollama_json(
    llm: ChatOllama,
    system_prompt: str,
    user_prompt: str,
    retries: int = 1,
    sleep: float = 0.3,
) -> str:
    """
    Pide JSON {"answer": "..."} y hace parse robusto.
    Si no logra parsear, devuelve texto crudo (sin newlines) como último recurso.
    """
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    last_exc: Optional[Exception] = None
    raw_txt: str = ""
    for _ in range(retries + 1):
        try:
            out = llm.invoke(msgs)
            raw_txt = (out.content or "").strip()
            # intento 1: parse directo
            try:
                data = json.loads(raw_txt)
            except Exception:
                # intento 2: extraer primer bloque {...}
                start = raw_txt.find("{")
                end = raw_txt.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError("Respuesta no contiene JSON parseable")
                data = json.loads(raw_txt[start:end+1])
            ans = GeneratedAnswer(**data).answer.strip()
            return ans
        except Exception as e:
            last_exc = e
            time.sleep(sleep)
    sys.stderr.write(f"[WARN] No pude parsear JSON de la respuesta. Error: {last_exc}\n")
    return raw_txt.replace("\n", " ").strip() or "Not enough information in the provided context."


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_CSV), help="Ruta a questions.csv (de 1-generar_preg.py)")
    ap.add_argument("--chunks", default=str(CHUNKS_JSONL), help="Ruta a out_chunks/chunks.jsonl")
    ap.add_argument("--out-jsonl", default=str(ANSWERS_JSONL), help="Salida JSONL")
    ap.add_argument("--out-csv", default=str(ANSWERS_CSV), help="Salida CSV")
    ap.add_argument("--model", default="llama3.1:8b", help="Modelo Ollama (ej: llama3.1:8b)")
    ap.add_argument("--ollama-base-url", default="http://localhost:11434", help="Base URL de Ollama")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=None, help="Limitar cantidad de preguntas")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep entre requests")
    args = ap.parse_args()

    # Asegurar carpetas
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Cargar preguntas
    dfq = pd.read_csv(args.questions)
    required = {"chunk_id", "question"}
    if not required.issubset(dfq.columns):
        raise RuntimeError(f"El CSV de preguntas debe contener columnas: {required}. Encontradas: {dfq.columns.tolist()}")

    if args.max_rows:
        dfq = dfq.head(args.max_rows).copy()

    # Cargar chunks en memoria (mapa)
    chunk_map = read_chunks_to_map(Path(args.chunks))
    if not chunk_map:
        raise RuntimeError(f"No se pudo cargar ningún chunk desde: {args.chunks}")

    # LLM
    llm = make_llm(args.model, args.ollama_base_url, args.temperature)

    # Generación
    records = []
    for i, row in dfq.iterrows():
        chunk_id = row.get("chunk_id")
        if pd.isna(chunk_id) or not str(chunk_id).strip():
            raise RuntimeError(f"Falta chunk_id en la fila {i} de {args.questions}")

        chunk_id = str(chunk_id).strip()
        chunk_obj = chunk_map.get(chunk_id)
        if not chunk_obj:
            raise RuntimeError(f"chunk_id={chunk_id} no existe en {args.chunks}. ¿Cambiaste los IDs o el archivo?")

        context = get_context_from_chunk(chunk_obj)
        if not context:
            raise RuntimeError(f"El chunk_id={chunk_id} no tiene texto ('page_content'/'text' vacío).")

        question = str(row.get("question") or "").strip()
        if not question:
            raise RuntimeError(f"Fila {i}: pregunta vacía en {args.questions}")

        user_prompt = ANSWER_PROMPT.format(context=context, question=question)
        system_prompt = "You are a precise medical assistant that answers only from the given context."

        answer = call_ollama_json(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retries=1,
            sleep=0.3,
        )

        records.append({
            "chunk_id": chunk_id,
            "question": question,
            "answer": answer,
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Guardar JSONL
    with Path(args.out_jsonl).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Guardar CSV
    pd.DataFrame(records).to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"✓ Generadas {len(records)} respuestas -> {args.out_jsonl}")
    print(f"✓ CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
