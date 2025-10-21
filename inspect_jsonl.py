#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_jsonl.py — utilitario para leer/filtrar/resumir chunks.jsonl

Ejemplos:
  # ver primeras 10 filas
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --limit 10

  # filtrar por droga (case-insensitive)
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --drug "ozempic" --limit 5

  # filtrar por sección canónica
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --section "CONTRAINDICACIONES" --limit 10

  # buscar texto que contenga "hipoglucemia"
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --contains "hipoglucemia"

  # usar regex en el texto
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --regex "(embarazo|lactancia)"

  # mostrar estadísticas
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --stats

  # mostrar texto completo
  python inspect_jsonl.py -f out_chunks/chunks.jsonl --drug "sertal" --section "POSOLOGÍA" --show-text --limit 3
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                yield rec
            except Exception as e:
                # no abortamos por una línea corrupta
                print(f"[WARN] línea {ln} inválida: {e}")

def record_matches(rec: Dict, drug: Optional[str], section: Optional[str],
                   contains: Optional[str], regex: Optional[str]) -> bool:
    md = rec.get("metadata", {}) or {}
    text = rec.get("text", "") or ""
    ok = True
    if drug:
        ok = ok and drug.lower() in (md.get("drug_name", "").lower())
    if section:
        ok = ok and section.lower() in (md.get("section_canonical", "").lower())
    if contains:
        ok = ok and (contains.lower() in text.lower())
    if regex:
        try:
            if not re.search(regex, text, re.IGNORECASE | re.MULTILINE):
                return False
        except re.error as e:
            print(f"[ERR] regex inválida: {e}")
            return False
    return ok

def preview_text(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def print_row(rec: Dict, show_text: bool, width: int = 140) -> None:
    md = rec.get("metadata", {}) or {}
    rid = rec.get("id")
    drug = md.get("drug_name")
    sec = md.get("section_canonical") or md.get("section_raw") or "UNKNOWN"
    txt = rec.get("text", "")
    if show_text:
        print("-----------------------------------------------------------------")
        print(f"- id={rid}\n  drug={drug}\n  section={sec}\n  text=\n{txt}\n")
    else:
        print(f"{rid:>16} | {drug:<25} | {sec:<28} | {preview_text(txt, n=width)}")

def do_stats(path: Path) -> None:
    by_drug = Counter()
    by_section = Counter()
    total = 0
    for rec in iter_jsonl(path):
        md = rec.get("metadata", {}) or {}
        by_drug[md.get("drug_name", "UNKNOWN")] += 1
        by_section[md.get("section_canonical", "UNKNOWN")] += 1
        total += 1

    print(f"\nTotal de chunks: {total}\n")
    print("Chunks por droga:")
    for k, v in by_drug.most_common():
        print(f"  - {k}: {v}")
    print("\nChunks por sección (canónica):")
    for k, v in by_section.most_common():
        print(f"  - {k}: {v}")

def main() -> int:
    ap = argparse.ArgumentParser(description="Leer/filtrar/resumir un JSONL de chunks para RAG.")
    ap.add_argument("-f", "--file", required=True, help="Ruta al .jsonl (p. ej. out_chunks/chunks.jsonl)")
    ap.add_argument("--drug", help="Filtro por metadata.drug_name (contiene, case-insensitive)")
    ap.add_argument("--section", help="Filtro por metadata.section_canonical (contiene, case-insensitive)")
    ap.add_argument("--contains", help="Filtro: texto del chunk contiene esta cadena (case-insensitive)")
    ap.add_argument("--regex", help="Filtro: regex aplicada al texto del chunk")
    ap.add_argument("--limit", type=int, default=20, help="Máximo de filas a mostrar (default 20)")
    ap.add_argument("--offset", type=int, default=0, help="Saltar las primeras N coincidencias")
    ap.add_argument("--show-text", action="store_true", help="Mostrar texto completo del chunk")
    ap.add_argument("--stats", action="store_true", help="Mostrar estadísticas en lugar de filas")
    args = ap.parse_args()

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"[ERR] No existe el archivo: {path}")
        return 2

    if args.stats:
        do_stats(path)
        return 0

    shown = 0
    skipped = 0
    for rec in iter_jsonl(path):
        if record_matches(rec, args.drug, args.section, args.contains, args.regex):
            if skipped < args.offset:
                skipped += 1
                continue
            print_row(rec, show_text=args.show_text)
            shown += 1
            if args.limit and shown >= args.limit:
                break

    if shown == 0:
        print("(sin resultados)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
