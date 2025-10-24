
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-retrieve.py

Compatibilidad de CLI con versiones anteriores de 4-retrieve.py:
- -q / --query
- -p / --persist-dir
- -c / --collection
- --k (n de resultados finales)
- --mmr (activa MMR)
- --fetch-k (n de candidatos iniciales para MMR/rerank)
- --answer (genera respuesta con LLM usando los contextos recuperados)
- --model (modelo LLM; por defecto 'llama3.1:latest' en Ollama)
- --answer-max-tokens (límite de tokens del LLM)
- --output (prefijo de archivos de salida; p.ej. 'answer' -> answer_answer.txt y answer_contexts.jsonl)
- --debug (imprime info adicional)
- --show-text-len (preview por fila)
- --device / --fp16 (para BGE-M3)
- --max-length (máx. tokens en el encoder de BGE-M3)

Requisitos:
  pip install "chromadb>=0.5.0" "FlagEmbedding>=1.2.11" "tqdm" "rich" "orjson" "requests"
  (Ollama opcional para --answer: https://ollama.ai, corre en localhost:11434)

Ejemplo (manteniendo tu forma de llamar):
  python 4-retrieve-bge-m3-compat.py \
    -q "¿Qué forma tienen los comprimidos de Enalapril?" \
    -p chroma/prospectos -c prospectos_bgem3 \
    --k 8 --mmr --fetch-k 40 --answer --output answer --debug
  
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Any

import chromadb
from FlagEmbedding import BGEM3FlagModel
from rich.console import Console
from rich.table import Table

console = Console()

def json_dumps(obj: Any) -> str:
    try:
        import orjson as fastjson
        return fastjson.dumps(obj).decode("utf-8")
    except Exception:
        return json.dumps(obj, ensure_ascii=False)

def format_rows(ids, docs, mds, dists) -> List[Tuple[str, float, str, dict]]:
    rows = []
    for i in range(len(ids)):
        rows.append((ids[i], dists[i], docs[i], mds[i]))
    return rows

def rerank_with_bge(query: str, rows: List[Tuple[str, float, str, dict]], device: str = "cpu", fp16: bool = False):
    """Reordena usando bge-reranker-v2-m3 (cross-encoder)."""
    try:
        from FlagEmbedding import FlagReranker
    except Exception:
        console.print("[yellow]Reranker no disponible (instala FlagEmbedding>=1.2.11). Continuo sin rerank.[/yellow]")
        return rows

    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=fp16, device=device)
    pairs = [(query, doc) for (_id, dist, doc, md) in rows]
    scores = reranker.compute_score(pairs, batch_size=64)
    scored = list(zip(scores, rows))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for (s, r) in scored]

def call_ollama(model: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Llama al endpoint local de Ollama (http://localhost:11434/api/generate).
    Devuelve el texto generado. Si falla, devuelve cadena vacía.
    """
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.ok:
            data = resp.json()
            return data.get("response", "").strip()
        else:
            console.print(f"[yellow]Ollama devolvió {resp.status_code}. Respuesta: {resp.text[:200]}[/yellow]")
            return ""
    except Exception as e:
        console.print(f"[yellow]No pude llamar a Ollama: {e}[/yellow]")
        return ""

def build_answer_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Crea un prompt conciso con citas referenciadas por ID."""
    ctx_lines = []
    for i, ctx in enumerate(contexts, start=1):
        preview = ctx["document"]
        if len(preview) > 900:
            preview = preview[:900] + "…"
        ctx_lines.append(f"[{i}] ID={ctx['id']}  METADATA={ctx.get('metadata', {})}\n{preview}")
    ctx_block = "\n\n".join(ctx_lines)
    prompt = f"""Eres un asistente experto en prospectos médicos. Responde con precisión y SIN inventar.
Si no hay información suficiente, di explícitamente "No encontrado en los documentos".
Cita las fuentes al final como [1], [2], etc.

Pregunta: {query}

Documentos recuperados (recorta tu respuesta solo a lo aquí citado):
{ctx_block}

Respuesta concisa y con citas:"""
    return prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", type=str, required=True, help="Consulta en lenguaje natural")
    ap.add_argument("-p", "--persist-dir", type=str, required=True, help="Directorio de persistencia de Chroma")
    ap.add_argument("-c", "--collection", type=str, default="prospectos", help="Nombre de la colección")
    ap.add_argument("--k", type=int, default=8, help="Cantidad de resultados finales")
    ap.add_argument("--fetch-k", type=int, default=50, help="Cantidad inicial (MMR o preranking)")
    ap.add_argument("--mmr", action="store_true", help="Activar MMR (diversidad)")
    ap.add_argument("--device", type=str, default="cuda" if os.environ.get("USE_CUDA") == "1" else "cpu",
                    help="cpu | cuda")
    ap.add_argument("--fp16", action="store_true", help="Usar FP16 si hay soporte (GPU recomendada)")
    ap.add_argument("--max-length", type=int, default=8192, help="Longitud máx. de tokens del encoder")
    ap.add_argument("--show-text-len", type=int, default=220, help="Caracteres de preview por fila")
    ap.add_argument("--rerank", action="store_true", help="(Opcional) Reordenar con bge-reranker-v2-m3")
    ap.add_argument("--answer", action="store_true", help="(Opcional) Generar respuesta con LLM (Ollama)")
    ap.add_argument("--model", type=str, default="llama3.1:latest", help="Modelo LLM para --answer (Ollama)")
    ap.add_argument("--answer-max-tokens", type=int, default=512, help="Límite de tokens para la respuesta")
    ap.add_argument("--output", type=str, default=None, help="Prefijo de salida (p.ej., 'answer')")
    ap.add_argument("--debug", action="store_true", help="Imprime detalles para diagnóstico")
    args = ap.parse_args()

    # 1) Cliente y colección
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=args.collection)
    console.print(f"[green]✓ Colección[/green]: {args.collection}  [green]persist_dir[/green]: {args.persist_dir}")

    # 2) Modelo BGEM3 (queries)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=args.fp16, device=args.device)

    encq = model.encode_queries([args.query], max_length=args.max_length)
    qvec = encq["dense_vecs"][0]

    # 3) Consulta a Chroma (con o sin MMR)
    query_kwargs = dict(n_results=args.k)
    if args.mmr:
        query_kwargs.update({"mmr": True, "n_results": args.k, "fetch_k": max(args.fetch_k, args.k)})
    else:
        query_kwargs.update({"n_results": max(args.k, args.fetch_k)})

    res = collection.query(query_embeddings=[qvec], **query_kwargs)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    mds  = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # en espacio cosine si la colección fue creada así

    rows = format_rows(ids, docs, mds, dists)

    # 4) Rerank opcional (mantiene compatibilidad si decides activarlo en tus pruebas)
    if args.rerank and rows:
        rows = rerank_with_bge(args.query, rows, device=args.device, fp16=args.fp16)

    # 5) Mostrar tabla en consola
    table = Table(title=f"Resultados para: {args.query}")
    table.add_column("#", style="bold")
    table.add_column("ID")
    table.add_column("Dist/Score", justify="right")
    table.add_column("Preview")
    table.add_column("Metadata")

    for i, (rid, dist, doc, md) in enumerate(rows[:args.k], start=1):
        preview = (doc[:args.show_text_len] + "…") if len(doc) > args.show_text_len else doc
        table.add_row(str(i), str(rid), f"{dist:.4f}", preview.replace("\n", " "), str(md))

    console.print(table)

    # 6) “Bandera” de relevant_fragment (heurstica simple)
    relevant = False
    threshold = 0.2  # menor distancia (cosine) implica más similitud; ajusta según distribución
    if dists:
        # algunos backends reportan similitud en lugar de distancia; manejamos robustamente
        # asumimos que si los valores > 1.2 es similitud; si < 1.0 es distancia.
        vals = [v for v in dists if v is not None]
        if vals and max(vals) > 1.2:
            # parece similitud -> invertimos criterio
            relevant = max(vals) >= 0.5
        else:
            relevant = min(vals) <= threshold
    tag = "[OK]" if relevant else "[WARN]"
    console.print(f"{tag} relevant_fragment: {relevant}  (heurística con threshold={threshold})")

    # 7) Output opcional (contexts + answer)
    if args.output:
        ctxs = []
        for rid, dist, doc, md in rows[:max(args.k, 20)]:
            ctxs.append({"id": rid, "distance_or_score": dist, "document": doc, "metadata": md})

        # contexts.jsonl
        ctx_path = f"{args.output}_contexts.jsonl"
        with open(ctx_path, "w", encoding="utf-8") as f:
            for c in ctxs:
                f.write(json_dumps(c) + "\n")
        console.print(f"[green]✓ Guardado[/green] contextos en: {ctx_path}")

        if args.answer:
            prompt = build_answer_prompt(args.query, ctxs[:args.k])
            answer = call_ollama(args.model, prompt, max_tokens=args.answer_max_tokens)
            if not answer:
                answer = "No pude generar respuesta con el LLM configurado."
            ans_path = f"{args.output}_answer.txt"
            with open(ans_path, "w", encoding="utf-8") as f:
                f.write(answer)
            console.print(f"[green]✓ Guardado[/green] answer en: {ans_path}")

if __name__ == "__main__":
    main()
