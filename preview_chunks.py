#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_chunks.py — Muestra (doc_name, drug_name, sección, primeras N palabras del chunk)
de un JSONL de chunks (formato 2-transform.py).

Uso:
  python preview_chunks.py -f chunks.jsonl
  python preview_chunks.py -f chunks.jsonl --n-words 12 --limit 50
  python preview_chunks.py -f chunks.jsonl --drug ozempic --section "POSOLOGÍA" --n-words 8
  python preview_chunks.py -f chunks.jsonl --csv > preview.csv
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional

WORD_RX = re.compile(r"[A-Za-zÁÉÍÓÚÜáéíóúüñÑ0-9]+(?:[-’'][A-Za-zÁÉÍÓÚÜáéíóúüñÑ0-9]+)*")

def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] línea {ln} inválida: {e}", file=sys.stderr)

def first_words(s: str, n: int) -> str:
    tokens = WORD_RX.findall(s or "")
    if not tokens:
        return ""
    out = " ".join(tokens[:n])
    if len(tokens) > n:
        out += "…"
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Previsualiza nombre de archivo, droga, sección y primeras N palabras del chunk.")
    ap.add_argument("-f", "--file", required=True, help="Ruta al JSONL (p. ej., out_chunks/chunks.jsonl)")
    ap.add_argument("--n-words", type=int, default=10, help="Cantidad de palabras a mostrar del texto (default 10)")
    ap.add_argument("--limit", type=int, default=0, help="Máximo de filas a mostrar (0 = sin límite)")
    ap.add_argument("--offset", type=int, default=0, help="Saltar las primeras N coincidencias")
    ap.add_argument("--drug", help="Filtro por metadata.drug_name (contiene, case-insensitive)")
    ap.add_argument("--section", help="Filtro por section_canonical (contiene, case-insensitive)")
    ap.add_argument("--contains", help="Filtro: texto del chunk contiene (case-insensitive)")
    ap.add_argument("--csv", action="store_true", help="Salida en CSV (en vez de tabla)")
    args = ap.parse_args()

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"[ERR] No existe el archivo: {path}", file=sys.stderr)
        return 2

    shown = 0
    skipped = 0

    if not args.csv:
        print(f"{'doc_name':<40} | {'drug_name':<18} | {'section':<34} | preview")
        print("-"*120)

    for rec in iter_jsonl(path):
        md = rec.get("metadata", {}) or {}
        text = rec.get("text", "") or ""

        doc_name = md.get("doc_name") or (md.get("doc_id") or "UNKNOWN")
        drug = md.get("drug_name") or "UNKNOWN"
        section = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"

        # filtros
        if args.drug and args.drug.lower() not in str(drug).lower():
            continue
        if args.section and args.section.lower() not in str(section).lower():
            continue
        if args.contains and args.contains.lower() not in text.lower():
            continue

        if skipped < args.offset:
            skipped += 1
            continue

        preview = first_words(text, args.n_words)

        if args.csv:
            # CSV simple: escapamos comas con comillas si hace falta
            def esc(x: str) -> str:
                x = x.replace('"', '""')
                return f'"{x}"'
            print(f"{esc(doc_name)},{esc(str(drug))},{esc(str(section))},{esc(preview)}")
        else:
            print(f"{doc_name:<40} | {str(drug)[:18]:<18} | {str(section)[:34]:<34} | {preview}")

        shown += 1
        if args.limit and shown >= args.limit:
            break

    if shown == 0:
        msg = "Sin resultados con los filtros aplicados." if (args.drug or args.section or args.contains) else "Sin registros."
        print(msg, file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
