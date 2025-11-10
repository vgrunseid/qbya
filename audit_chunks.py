#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re, math, statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def approx_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))

def sliding_overlap_ratio(a: str, b: str, win_words: int = 80) -> float:
    # Estima solape entre el final de a y el inicio de b, en palabras
    aw = (a or "").split()
    bw = (b or "").split()
    tail = " ".join(aw[-win_words:])
    head = " ".join(bw[:win_words])
    if not tail or not head:
        return 0.0
    # mide coincidencia por longitud de LCS aproximada (intersección de shingles)
    def shingles(s, n=5):
        ws = s.split()
        return {" ".join(ws[i:i+n]) for i in range(max(0, len(ws)-n+1))}
    A, B = shingles(tail), shingles(head)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def is_bullet_like(line: str) -> bool:
    return bool(re.match(r"^\s*[-•*]\s+\S", line))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="Ruta a chunks.jsonl")
    ap.add_argument("--overlap-win", type=int, default=80, help="Ventana aprox de solape (palabras)")
    ap.add_argument("--long-th", type=int, default=900, help="Umbral de línea muy larga (caracteres)")
    args = ap.parse_args()

    path = Path(args.input)
    assert path.exists(), f"No existe {path}"

    sizes = []
    per_section = Counter()
    per_doc = defaultdict(int)
    long_lines = 0
    bullet_lines = 0
    overlaps = []

    # Para medir solape: recordar último chunk por (doc_id, section)
    last_chunk_by_key = {}

    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get("text","")
            md = obj.get("metadata", {})
            doc_id = md.get("doc_id") or md.get("document_id") or "UNK"
            sec = md.get("section_canonical") or "UNKNOWN"
            chunk_idx = md.get("chunk_index", None)

            # tamaños
            tok = approx_token_count(text)
            sizes.append(tok)
            per_section[sec] += 1
            per_doc[doc_id] += 1
            total += 1

            # bullets y líneas largas
            for ln in text.splitlines():
                if is_bullet_like(ln): bullet_lines += 1
                if len(ln) >= args.long_th: long_lines += 1

            # solape efectivo entre chunks consecutivos en misma sección
            key = (doc_id, sec)
            if key in last_chunk_by_key and isinstance(chunk_idx, int):
                prev = last_chunk_by_key[key]
                if chunk_idx == prev["idx"] + 1:
                    r = sliding_overlap_ratio(prev["text"], text, args.overlap_win)
                    overlaps.append(r)
            if isinstance(chunk_idx, int):
                last_chunk_by_key[key] = {"idx": chunk_idx, "text": text}

    print(f"→ Total de chunks: {total}")
    if sizes:
        print(f"→ Tokens aprox / chunk: media={stats.mean(sizes):.1f}  mediana={stats.median(sizes):.1f}  p90={stats.quantiles(sizes, n=10)[-1]:.0f}")
        print(f"   min={min(sizes)}  max={max(sizes)}")
        # histograma simple
        bins = [0,120,240,360,480,600,720,840,10000]
        hist = Counter()
        for s in sizes:
            for i in range(len(bins)-1):
                if bins[i] <= s < bins[i+1]:
                    hist[(bins[i], bins[i+1])] += 1
                    break
        print("→ Histograma aprox (tokens):")
        for (a,b), c in sorted(hist.items()):
            print(f"   {a:>3}-{b-1:<3} : {c}")

    print(f"→ Chunks por sección (top 12):")
    for sec, c in per_section.most_common(12):
        print(f"   {sec:<35} {c}")

    print(f"→ Docs indexados: {len(per_doc)}  |  chunks/doc (mediana)={stats.median(per_doc.values()) if per_doc else 0}")
    print(f"→ Líneas tipo bullet detectadas: {bullet_lines}")
    print(f"→ Líneas MUY largas (>{args.long_th} chars): {long_lines}")

    if overlaps:
        print(f"→ Solape efectivo (Jaccard de shingles) entre consecutivos:")
        print(f"   media={stats.mean(overlaps):.3f}  mediana={stats.median(overlaps):.3f}  p90={stats.quantiles(overlaps, n=10)[-1]:.3f}")
        low = sum(1 for r in overlaps if r < 0.02)
        print(f"   ~{low}/{len(overlaps)} pares con solape casi nulo (<0.02)")

if __name__ == "__main__":
    main()
