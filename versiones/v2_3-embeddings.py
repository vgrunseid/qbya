#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-retrieve.py — Retrieve con mismos flags/salida que tu versión previa.
Único cambio: el embedding de la PREGUNTA ahora se calcula con BGE-M3 (1024-d).

Ejemplo:
  python 4-retrieve.py \
    -q "¿Qué forma tienen los comprimidos de Enalapril?" \
    -p chroma/prospectos -c prospectos_bgem3 \
    --k 8 --mmr --fetch-k 40 --answer --output answer --debug
"""

import argparse
import json
import math
import os
from typing import List, Tuple, Dict, Any

import chromadb
from rich.console import Console
from rich.table import Table

console = Console()

# --------------------------- Utils (igual que antes) ---------------------------

def json_dumps(obj: Any) -> str:
    try:
        import orjson as fastjson
        return fastjson.dumps(obj).decode("utf-8")
    except Exception:
        return json.dumps(obj, ensure_ascii=False)

def _tolist_if_ndarray(x):
    return x.tolist() if hasattr(x, "tolist") else x

def cosine_sim(a, b) -> float:
    a = _tolist_if_ndarray(a); b = _tolist_if_ndarray(b)
    num = 0.0; da = 0.0; db = 0.0
    for x, y in zip(a, b):
        num += x * y; da += x * x; db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / ((da ** 0.5) * (db ** 0.5))

def mmr_select(query_vec, doc_vecs, k: int, lamb: float = 0.5):
    # (sin cambios funcionales; sólo robusto a ndarrays)
    if hasattr(doc_vecs, "tolist"): doc_vecs = doc_vecs.tolist()
    if hasattr(query_vec, "tolist"): query_vec = query_vec.tolist()
    if doc_vecs is None or len(doc_vecs) == 0:
        return []
    selected = []; candidates = list(range(len(doc_vecs)))
    sim_q = [cosine_sim(query_vec, v) for v in doc_vecs]
    while len(selected) < min(k, len(doc_vecs)):
        best_idx = None; best_score = float("-inf")
        for i in candidates:
            if not selected:
                score = sim_q[i]
            else:
                max_sim_to_S = max(cosine_sim(doc_vecs[i], doc_vecs[j]) for j in selected)
                score = lamb * sim_q[i] - (1 - lamb) * max_sim_to_S
            if score > best_score:
                best_score = score; best_idx = i
        selected.append(best_idx); candidates.remove(best_idx)
    return selected

def rerank_with_bge(query: str, rows: List[Tuple[str, float, str, dict]], device: str = "cpu", fp16: bool = False):
    try:
        from FlagEmbedding import FlagReranker
    except Exception:
        return rows
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=fp16, device=device)
    pairs = [(query, doc) for (_id, dist, doc, md) in rows]
    scores = reranker.compute_score(pairs, batch_size=64)
    scored = list(zip(scores, rows))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for (s, r) in scored]

def call_ollama(model: str, prompt: str, max_tokens: int = 512) -> str:
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.ok:
            data = resp.json()
            return data.get("response", "").strip()
        return ""
    except Exception:
        return ""

def build_answer_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for i, ctx in enumerate(contexts, start=1):
        preview = ctx["document"]
        if len(preview) > 900:
            preview = preview[:900] + "…"
        ctx_lines.append(f"[{i}] ID={ctx['id']}  METADATA={ctx.get('metadata', {})}\n{preview}")
    ctx_block = "\n\n".join(ctx_lines)
    return f"""Eres un asistente experto en prospectos médicos. Responde con precisión y SIN inventar.
Si no hay información suficiente, di explícitamente "No encontrado en los documentos".
Cita las fuentes al final como [1], [2], etc.

Pregunta: {query}

Documentos recuperados (recorta tu respuesta solo a lo aquí citado):
{ctx_block}

Respuesta concisa y con citas:"""

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", type=str, required=True)
    ap.add_argument("-p", "--persist-dir", type=str, required=True)
    ap.add_argument("-c", "--collection", type=str, default="prospectos")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fetch-k", type=int, default=50)
    ap.add_argument("--mmr", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if os.environ.get("USE_CUDA") == "1" else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--show-text-len", type=int, default=220)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--model", type=str, default="llama3.1:latest")
    ap.add_argument("--answer-max-tokens", type=int, default=512)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # 1) Cliente y colección (igual que antes)
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=args.collection)
    console.print(f"[green]✓ Colección[/green]: {args.collection}  [green]persist_dir[/green]: {args.persist_dir}")

    # 2) *** ÚNICO CAMBIO: embedding de la PREGUNTA con BGE-M3 (1024-d) ***
    # === BEGIN MINIMAL PATCH: BGE-M3 para la QUERY (no tocar nada más) ===
    from FlagEmbedding import BGEM3FlagModel
    _device = getattr(args, "device", "cpu")
    _fp16   = bool(getattr(args, "fp16", False))
    _bge_q  = BGEM3FlagModel("BAAI/bge-m3", use_fp16=_fp16, device=_device)
    _maxlen = getattr(args, "max_length", 8192)
    _encq   = _bge_q.encode_queries([args.query], max_length=_maxlen)
    qvec    = _encq["dense_vecs"][0]   # <-- 1024-d, compatible con colección indexada con BGE-M3
    # === END MINIMAL PATCH ===

    # 3) Query a Chroma (sin cambios)
    base_n = max(args.k, args.fetch_k)
    res = collection.query(query_embeddings=[qvec], n_results=base_n)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    mds  = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids   = _tolist_if_ndarray(ids); docs  = _tolist_if_ndarray(docs)
    mds   = _tolist_if_ndarray(mds); dists = _tolist_if_ndarray(dists)

    rows = []
    n = min(len(ids), len(docs), len(mds), len(dists))
    for i in range(n):
        rows.append((ids[i], dists[i], docs[i], mds[i]))
    selected_rows = rows

    # 4) MMR (sin cambiar tu lógica; si tu versión no soporta mmr en query, ya lo tenías resuelto o desactivado)
    if args.mmr and rows:
        # intentar embeddings guardados para MMR
        doc_vecs = None
        try:
            got = collection.get(ids=ids, include=["embeddings"])
            emb_list = got.get("embeddings", None)
            if emb_list and len(emb_list) == len(ids):
                doc_vecs = emb_list
        except Exception:
            doc_vecs = None
        # si no hay, podés mantener tu fallback previo (si ya lo tenías) o comentar esta parte
        if doc_vecs is not None:
            doc_vecs = _tolist_if_ndarray(doc_vecs)
            qvec_ = _tolist_if_ndarray(qvec)
            sel_idx = mmr_select(qvec_, doc_vecs, k=args.k, lamb=0.5)
            selected_rows = [rows[i] for i in sel_idx]
        # (si querés desactivar MMR temporalmente dejá: selected_rows = rows)

    # 5) Rerank (igual que antes)
    if args.rerank and selected_rows:
        selected_rows = rerank_with_bge(args.query, selected_rows, device=args.device, fp16=args.fp16)

    # 6) Mostrar tabla (igual que antes)
    table = Table(title=f"Resultados para: {args.query}")
    table.add_column("#", style="bold")
    table.add_column("ID")
    table.add_column("Dist/Score", justify="right")
    table.add_column("Preview")
    table.add_column("Metadata")
    for i, (rid, dist, doc, md) in enumerate(selected_rows[:args.k], start=1):
        preview = (doc[:args.show_text_len] + "…") if len(doc) > args.show_text_len else doc
        table.add_row(str(i), str(rid), f"{float(dist):.4f}", preview.replace("\n", " "), str(md))
    console.print(table)

    # 7) Bandera relevante (igual que antes)
    relevant = False
    threshold = 0.2
    vals = [v for v in dists if v is not None]
    if vals:
        if max(vals) > 1.2:      # similitud
            relevant = max(vals) >= 0.5
        else:                    # distancia
            relevant = min(vals) <= threshold
    tag = "[OK]" if relevant else "[WARN]"
    console.print(f"{tag} relevant_fragment: {relevant}  (heurística con threshold={threshold})")

    # 8) Salidas (igual que antes)
    if args.output:
        ctxs = []
        for rid, dist, doc, md in selected_rows[:max(args.k, 20)]:
            ctxs.append({"id": rid, "distance_or_score": float(dist), "document": doc, "metadata": md})
        ctx_path = f"{args.output}_contexts.jsonl"
        with open(ctx_path, "w", encoding="utf-8") as f:
            for c in ctxs:
                f.write(json_dumps(c) + "\n")
        console.print(f"[green]✓ Guardado[/green] contextos en: {ctx_path}")
        if args.answer:
            prompt = build_answer_prompt(args.query, ctxs[:args.k])
            answer = call_ollama(args.model, prompt, max_tokens=args.answer_max_tokens) or \
                     "No pude generar respuesta con el LLM configurado."
            ans_path = f"{args.output}_answer.txt"
            with open(ans_path, "w", encoding="utf-8") as f:
                f.write(answer)
            console.print(f"[green]✓ Guardado[/green] answer en: {ans_path}")

if __name__ == "__main__":
    main()
