#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-pdf_to_txt.py — Convertir PDFs a TXT (UTF-8) usando:
  - MarkItDown (rápido, muy bueno en muchos casos)
  - PDFMiner (layout-aware; mejora lectura en PDFs multicolumna)
  - Docling (con OCR opcional; útil para escaneados)

Incluye limpieza avanzada:
  - Unir palabras cortadas por guion al final de línea ("higie-\nne" → "higiene")
  - Desenrollar saltos de línea por maquetado con heurística "smart" (mantiene títulos/listas)

Usos típicos:
  # 1) Auto (MarkItDown → PDFMiner → Docling) + reflujo de párrafos
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine auto --unwrap-lines

  # 2) Forzar PDFMiner (suele arreglar lecturas de columnas)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine pdfminer --unwrap-lines

  # 3) Forzar Docling sin OCR (PDFs digitales)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine docling --ocr-mode never --unwrap-lines

  # 4) Docling con OCR automático (si el texto queda corto, reintenta OCR full-page)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine docling --ocr spa --ocr-mode auto --unwrap-lines

Notas:
  - markitdown[pdf] para usar MarkItDown con PDFs
  - pdfminer.six para el motor PDFMiner
  - docling para Docling (y Tesseract si vas a usar OCR)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import unicodedata
import subprocess
import platform
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional, List

# ---------- SSL fix (certifi) ----------
def _bootstrap_ssl():
    try:
        import certifi  # type: ignore
        cacert = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", cacert)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cacert)
    except Exception:
        pass

_bootstrap_ssl()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IN_DIR = BASE_DIR / "pdfs_crudos"
DEFAULT_OUT_DIR = BASE_DIR / "out_txt"

# ---------- utilidades ----------
def iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
        return
    if not path.exists():
        return
    for p in sorted(path.rglob("*.pdf")):
        yield p

def _maybe_set_tessdata_prefix():
    if platform.system() != "Darwin":
        return
    if os.getenv("TESSDATA_PREFIX"):
        return
    for p in ("/opt/homebrew/share/tessdata", "/usr/local/share/tessdata"):
        if os.path.isdir(p):
            os.environ["TESSDATA_PREFIX"] = p
            break

def _tesseract_has_lang(lang: str) -> bool:
    try:
        out = subprocess.check_output(
            ["tesseract", "--list-langs"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return any(line.strip() == lang for line in out.splitlines())
    except Exception:
        return False

def _normalize_ocr_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    lang = s.strip().lower()
    if lang in ("es", "es-419", "es_ar", "es-ar", "spa"):
        return "spa"
    if lang == "auto":
        return "auto"
    return lang or None

# ---------- limpieza de texto ----------
def unhyphenate_linebreaks(s: str) -> str:
    s = re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])-\s*\n\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])', r'\1\2', s)
    return s

def unwrap_paragraphs_smart(s: str) -> str:
    def is_list_like(line: str) -> bool:
        return bool(re.match(r'^\s*(?:[-*•‣·]|[0-9]+[.)]|[A-Za-z]\))\s+', line))

    def is_heading(line: str) -> bool:
        t = line.strip()
        if not t or len(t) > 120:
            return False
        words = t.split()
        caps_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
        title_case = (len(words) <= 10) and all(w[:1].isupper() for w in words if w)
        md_heading = t.startswith("#")
        return caps_ratio > 0.6 or title_case or md_heading

    def ends_hard(line: str) -> bool:
        return bool(re.search(r'[.!?¿¡:;»”)\]]\s*$', line))

    def looks_wrapped(prev: str, cur: str) -> bool:
        if is_list_like(cur) or is_heading(cur):
            return False
        if ends_hard(prev):
            return False
        if re.match(r'^\s*[,\.;:)\]]', cur):
            return True
        m = re.match(r'^\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])', cur)
        if m and m.group(1).islower():
            return True
        if re.search(
            r'\b(?:de|la|el|y|o|u|que|para|en|con|por|del|al|un|una|unos|unas|los|las|se|su|sus|lo|le|les|si|no|como|pero|más|menos)$',
            prev.strip(), re.IGNORECASE
        ):
            return True
        L1, L2 = len(prev.strip()), len(cur.strip())
        if 40 <= L1 <= 140 and 40 <= L2 <= 140:
            return True
        return False

    lines = s.splitlines()
    out = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if not prev.strip() or not line.strip():
            out.append(line)
            continue
        if looks_wrapped(prev, line):
            out[-1] = prev.rstrip() + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)

def clean_text(s: str, unwrap_mode: str = "smart", unhyphenate: bool = True) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if unhyphenate:
        s = unhyphenate_linebreaks(s)
    if unwrap_mode == "simple":
        s = re.sub(r'(?<![.!?:;»”)\]])\n(?!\n)', ' ', s)
    elif unwrap_mode == "smart":
        s = unwrap_paragraphs_smart(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------- motores: MarkItDown ----------
def convert_with_markitdown(pdf_path: Path) -> str:
    try:
        from markitdown import MarkItDown  # type: ignore
    except Exception as e:
        raise RuntimeError("MarkItDown no está instalado. pip install 'markitdown[pdf]'") from e
    md = MarkItDown()
    try:
        res = md.convert(pdf_path)
        text = getattr(res, "text_content", None) or getattr(res, "markdown", None) or str(res)
        return text
    except Exception as e:
        msg = str(e)
        if "MissingDependencyException" in msg:
            raise RuntimeError("MarkItDown reconoce PDF pero faltan deps: pip install 'markitdown[pdf]'") from e
        raise

# ---------- motores: PDFMiner (layout-aware, dos columnas) ----------
def _looks_two_columns(x_positions: List[float], page_width: float) -> bool:
    if not x_positions or not page_width:
        return False
    xs = sorted(x_positions)
    # Heurística: si hay masa a la izquierda (<40%) y a la derecha (>60%)
    left = sum(1 for x in xs if x < page_width * 0.40)
    right = sum(1 for x in xs if x > page_width * 0.60)
    return (left > 5 and right > 5)

def convert_with_pdfminer(
    pdf_path: Path,
    two_columns: str = "auto",  # auto | yes | no
    line_margin: float = 0.2,
    char_margin: float = 2.0,
    word_margin: float = 0.1,
) -> str:
    try:
        from pdfminer.high_level import extract_pages  # type: ignore
        from pdfminer.layout import LTTextContainer, LTTextLine, LTTextLineHorizontal, LAParams  # type: ignore
    except Exception as e:
        raise RuntimeError("pdfminer.six no está instalado. pip install pdfminer.six") from e

    laparams = LAParams(
        all_texts=True,
        line_margin=line_margin,
        char_margin=char_margin,
        word_margin=word_margin,
        detect_vertical=False,
    )

    pages_out: List[str] = []

    for page in extract_pages(str(pdf_path), laparams=laparams):
        # ancho de página
        try:
            page_width = getattr(page, "width", None) or (page.bbox[2] - page.bbox[0])
        except Exception:
            page_width = 0.0

        lines: List[Tuple[float, float, float, float, str]] = []  # (x0, y0, x1, y1, text)
        for elem in page:
            if isinstance(elem, LTTextContainer):
                for tl in elem:
                    if isinstance(tl, (LTTextLine, LTTextLineHorizontal)):
                        txt = tl.get_text()
                        if txt and txt.strip():
                            x0, y0, x1, y1 = tl.bbox
                            lines.append((x0, y0, x1, y1, txt))

        if not lines:
            pages_out.append("")
            continue

        # ¿dos columnas?
        x0s = [x0 for (x0, _, _, _, _) in lines]
        is_two = (two_columns == "yes") or (two_columns == "auto" and _looks_two_columns(x0s, page_width))

        page_text_parts: List[str] = []
        if is_two:
            # umbral de partición por mediana de x0
            split_x = sorted(x0s)[len(x0s) // 2]
            left = [t for t in lines if t[0] <= split_x]
            right = [t for t in lines if t[0] > split_x]
            # ordenar de arriba hacia abajo (y1 desc), luego x0 asc
            left.sort(key=lambda r: (-r[3], r[0]))
            right.sort(key=lambda r: (-r[3], r[0]))
            page_text_parts.append("\n".join(t for *_, t in left))
            page_text_parts.append("\n".join(t for *_, t in right))
        else:
            # una columna: ordenar por y1 desc, x0 asc
            lines.sort(key=lambda r: (-r[3], r[0]))
            page_text_parts.append("\n".join(t for *_, t in lines))

        pages_out.append("\n".join(page_text_parts))

    raw = "\n\n".join(pages_out)
    return raw

# ---------- motores: Docling ----------
def get_docling_converter_no_ocr():
    from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
    pdf_opts = PdfPipelineOptions(do_ocr=False)
    fmt = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    return DocumentConverter(format_options=fmt)

def get_docling_converter_with_ocr(ocr_engine: str, ocr_lang: Optional[str], force_full: bool):
    from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )  # type: ignore

    pdf_opts = PdfPipelineOptions(do_ocr=True)
    lang = _normalize_ocr_lang(ocr_lang)
    if lang == "auto" and _tesseract_has_lang("spa"):
        lang_list = ["spa"]
    else:
        lang_list = [lang] if lang else ["auto"]

    if ocr_engine == "tesseract-cli":
        pdf_opts.ocr_options = TesseractCliOcrOptions(
            lang=lang_list,
            force_full_page_ocr=bool(force_full),
        )
    elif ocr_engine == "ocrmac":
        try:
            from docling.datamodel.pipeline_options import OcrMacOptions  # type: ignore
            pdf_opts.ocr_options = OcrMacOptions(lang=lang_list if lang else None)
        except Exception as e:
            raise RuntimeError("Docling no tiene OcrMacOptions. Actualizá docling o usá tesseract-cli.") from e
    else:
        raise ValueError(f"OCR engine desconocido: {ocr_engine}")

    fmt = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    return DocumentConverter(format_options=fmt)

def convert_with_docling(
    pdf_path: Path,
    ocr_mode: str,
    ocr_engine: str,
    ocr_lang: Optional[str],
    fallback_min_chars: int = 300,
) -> Tuple[str, Dict[str, Any]]:
    try:
        import docling  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError("Docling no está instalado. pip install docling") from e

    info = {"used_ocr": False, "fallback_triggered": False, "engine": None}

    def _run(conv) -> str:
        res = conv.convert(str(pdf_path))
        if res.document is None:
            raise RuntimeError(f"Docling falló: status={res.status}, errors={res.errors}")
        return res.document.export_to_text()

    if ocr_mode == "force":
        conv = get_docling_converter_with_ocr(ocr_engine, ocr_lang, force_full=True)
        txt = _run(conv)
        info.update({"used_ocr": True, "fallback_triggered": False, "engine": ocr_engine, "len": len(txt)})
        return txt, info

    if ocr_mode == "never" or not ocr_lang:
        conv = get_docling_converter_no_ocr()
        txt = _run(conv)
        info.update({"used_ocr": False, "fallback_triggered": False, "engine": None, "len": len(txt)})
        return txt, info

    conv_no = get_docling_converter_no_ocr()
    text_no = _run(conv_no)
    if len(text_no) >= fallback_min_chars:
        info.update({"used_ocr": False, "fallback_triggered": False, "engine": None, "len": len(text_no)})
        return text_no, info

    try:
        conv_ocr = get_docling_converter_with_ocr(ocr_engine, ocr_lang, force_full=True)
        text_ocr = _run(conv_ocr)
        info.update({"used_ocr": True, "fallback_triggered": True, "engine": ocr_engine, "len": len(text_ocr)})
        return text_ocr, info
    except Exception:
        info.update({"used_ocr": False, "fallback_triggered": True, "engine": None, "len": len(text_no), "ocr_error": True})
        return text_no, info

# ---------- wrapper principal ----------
def convert_pdf_to_text(
    pdf_path: Path,
    engine: str = "auto",              # auto | markitdown | pdfminer | docling
    # limpieza
    unwrap_mode: str = "smart",
    do_unwrap: bool = True,
    unhyphenate: bool = True,
    # pdfminer params
    two_columns: str = "auto",         # auto | yes | no
    line_margin: float = 0.2,
    char_margin: float = 2.0,
    word_margin: float = 0.1,
    # docling params
    ocr_mode: str = "auto",
    ocr_engine: str = "tesseract-cli",
    ocr_lang: Optional[str] = None,
    fallback_min_chars: int = 300,
) -> Tuple[str, Dict[str, Any]]:
    if engine not in {"auto", "markitdown", "pdfminer", "docling"}:
        raise ValueError("engine debe ser: auto | markitdown | pdfminer | docling")

    def _post(text: str) -> str:
        return clean_text(
            text,
            unwrap_mode=(unwrap_mode if do_unwrap else "off"),
            unhyphenate=unhyphenate,
        )

    # --- AUTO: MarkItDown → PDFMiner → Docling
    if engine == "auto":
        # 1) MarkItDown
        try:
            raw = convert_with_markitdown(pdf_path)
            return _post(raw), {"engine": "markitdown", "used_ocr": False, "len": len(raw)}
        except Exception as e:
            print(f"[INFO] MarkItDown falló para {pdf_path.name}. Intento PDFMiner. Motivo: {e}", file=sys.stderr)
        # 2) PDFMiner
        try:
            raw = convert_with_pdfminer(
                pdf_path,
                two_columns=two_columns,
                line_margin=line_margin,
                char_margin=char_margin,
                word_margin=word_margin,
            )
            return _post(raw), {"engine": f"pdfminer({two_columns})", "used_ocr": False, "len": len(raw)}
        except Exception as e:
            print(f"[INFO] PDFMiner falló para {pdf_path.name}. Intento Docling. Motivo: {e}", file=sys.stderr)
        # 3) Docling
        if ocr_mode in ("auto", "force") and ocr_lang:
            if ocr_engine == "tesseract-cli":
                _maybe_set_tessdata_prefix()
                if shutil.which("tesseract") is None:
                    print("[WARN] No encontré 'tesseract' en PATH. (macOS: brew install tesseract) o cambiar a --ocr-engine ocrmac.", file=sys.stderr)
                else:
                    lang_chk = _normalize_ocr_lang(ocr_lang) or "spa"
                    if lang_chk != "auto" and not _tesseract_has_lang(lang_chk):
                        print(f"[WARN] Tesseract sin idioma '{lang_chk}'. Instalalo (p.ej. 'spa'). Sigo con fallback…", file=sys.stderr)
            elif ocr_engine == "ocrmac" and platform.system() != "Darwin":
                print("[WARN] --ocr-engine ocrmac solo en macOS. Cambio a tesseract-cli.", file=sys.stderr)
                ocr_engine = "tesseract-cli"

        raw, info = convert_with_docling(
            pdf_path=pdf_path,
            ocr_mode=ocr_mode,
            ocr_engine=ocr_engine,
            ocr_lang=ocr_lang,
            fallback_min_chars=fallback_min_chars,
        )
        return _post(raw), {"engine": f"docling/{ocr_engine}", **info}

    # --- MarkItDown
    if engine == "markitdown":
        raw = convert_with_markitdown(pdf_path)
        return _post(raw), {"engine": "markitdown", "used_ocr": False, "len": len(raw)}

    # --- PDFMiner
    if engine == "pdfminer":
        raw = convert_with_pdfminer(
            pdf_path,
            two_columns=two_columns,
            line_margin=line_margin,
            char_margin=char_margin,
            word_margin=word_margin,
        )
        return _post(raw), {"engine": f"pdfminer({two_columns})", "used_ocr": False, "len": len(raw)}

    # --- Docling
    if engine == "docling":
        if ocr_mode in ("auto", "force") and ocr_lang:
            if ocr_engine == "tesseract-cli":
                _maybe_set_tessdata_prefix()
                if shutil.which("tesseract") is None:
                    print("[WARN] No encontré 'tesseract' en PATH. Instalalo (macOS: brew install tesseract) o cambiá a --ocr-engine ocrmac.", file=sys.stderr)
                else:
                    lang_chk = _normalize_ocr_lang(ocr_lang) or "spa"
                    if lang_chk != "auto" and not _tesseract_has_lang(lang_chk):
                        print(f"[WARN] Tesseract no tiene '{lang_chk}'. Instalá (p.ej. 'spa'). Sigo con fallback…", file=sys.stderr)
            elif ocr_engine == "ocrmac" and platform.system() != "Darwin":
                print("[WARN] --ocr-engine ocrmac solo en macOS. Cambio a tesseract-cli.", file=sys.stderr)
                ocr_engine = "tesseract-cli"

        raw, info = convert_with_docling(
            pdf_path=pdf_path,
            ocr_mode=ocr_mode,
            ocr_engine=ocr_engine,
            ocr_lang=ocr_lang,
            fallback_min_chars=fallback_min_chars,
        )
        return _post(raw), {"engine": f"docling/{ocr_engine}", **info}

    raise RuntimeError("No se pudo convertir el PDF")

# ---------- CLI ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Convierte PDFs → TXT con MarkItDown / PDFMiner / Docling (con OCR).")
    ap.add_argument("-i", "--input", default=str(DEFAULT_IN_DIR), help=f"Archivo o carpeta de PDFs (default: {DEFAULT_IN_DIR})")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT_DIR), help=f"Carpeta de salida para .txt (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--engine", choices=["auto", "markitdown", "pdfminer", "docling"], default="auto",
                    help="Motor de conversión (auto=MarkItDown→PDFMiner→Docling).")

    # Limpieza
    ap.add_argument("--unwrap-lines", action="store_true", help="Desenrollar saltos de línea dentro de párrafos.")
    ap.add_argument("--unwrap-strategy", choices=["off", "simple", "smart"], default="smart",
                    help="Estrategia de desenrollado de líneas (default: smart).")
    ap.add_argument("--no-unhyphenate", action="store_true", help="No unir palabras cortadas con guion al final de línea.")

    # PDFMiner
    ap.add_argument("--pdfminer-cols", choices=["auto", "yes", "no"], default="auto",
                    help="Lectura de 2 columnas (auto|yes|no).")
    ap.add_argument("--pdfminer-line-margin", type=float, default=0.2, help="LAParams.line_margin (default 0.2)")
    ap.add_argument("--pdfminer-char-margin", type=float, default=2.0, help="LAParams.char_margin (default 2.0)")
    ap.add_argument("--pdfminer-word-margin", type=float, default=0.1, help="LAParams.word_margin (default 0.1)")

    # Docling / OCR
    ap.add_argument("--ocr", default=None, help="Idioma OCR (tesseract usa 'spa' para español; 'auto' para autodetección).")
    ap.add_argument("--ocr-mode", choices=["never", "auto", "force"], default="auto",
                    help="Cuándo aplicar OCR con Docling (default: auto).")
    ap.add_argument("--ocr-engine", choices=["tesseract-cli", "ocrmac"], default="tesseract-cli",
                    help="Motor OCR para Docling (default: tesseract-cli).")
    ap.add_argument("--min-chars-ocr", type=int, default=300,
                    help="Umbral de texto mínimo para activar OCR en modo auto (default 300).")

    ap.add_argument("--skip-existing", action="store_true", help="Saltar PDFs cuyo .txt ya existe.")

    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERR] La ruta de entrada no existe: {in_path}", file=sys.stderr)
        return 2

    pdfs = list(iter_pdfs(in_path))
    if not pdfs:
        print("[WARN] No se encontraron archivos .pdf en la entrada.", file=sys.stderr)
        return 3

    print(f"In : {in_path}")
    print(f"Out: {out_dir}")

    ok = fail = 0
    for i, pdf in enumerate(pdfs, 1):
        out_txt = out_dir / (pdf.stem + ".txt")
        if args.skip_existing and out_txt.exists():
            print(f"[{i}/{len(pdfs)}] ↷ {pdf.name} (saltado)")
            continue

        try:
            text, info = convert_pdf_to_text(
                pdf_path=pdf,
                engine=args.engine,
                unwrap_mode=args.unwrap_strategy,
                do_unwrap=args.unwrap_lines,
                unhyphenate=not args.no_unhyphenate,
                two_columns=args.pdfminer_cols,
                line_margin=args.pdfminer_line_margin,
                char_margin=args.pdfminer_char_margin,
                word_margin=args.pdfminer_word_margin,
                ocr_mode=args.ocr_mode,
                ocr_engine=args.ocr_engine,
                ocr_lang=args.ocr,
                fallback_min_chars=args.min_chars_ocr,
            )
            out_txt.write_text(text, encoding="utf-8")
            post = ""
            if str(info.get("engine","")).startswith("docling"):
                if info.get("fallback_triggered") and info.get("used_ocr"):
                    post = " (Docling + OCR fallback aplicado)"
                elif info.get("fallback_triggered") and not info.get("used_ocr"):
                    post = " (Docling: OCR fallback falló; quedó sin OCR)"
            print(f"[{i}/{len(pdfs)}] ✔ {pdf.name} → {out_txt.name} via {info.get('engine')}{post}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] ✖ {pdf.name}: {e}", file=sys.stderr)
            fail += 1

    print(f"Terminado. OK: {ok} | Errores: {fail} | Total: {len(pdfs)} → {out_dir}")
    return 0 if ok > 0 else 4

if __name__ == "__main__":
    raise SystemExit(main())
