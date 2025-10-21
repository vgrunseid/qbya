#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-pdf_to_txt.py — Convertir PDFs a TXT (UTF-8) usando MarkItDown (preferido) o Docling (con OCR opcional).
Incluye limpieza avanzada:
  - Unir palabras cortadas por guion al final de línea ("higie-\nne" → "higiene")
  - Desenrollar saltos de línea por maquetado con heurística "smart" (mantiene títulos/listas)

Usos típicos:
  # 1) Solo MarkItDown (por defecto, si está instalado)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine auto --unwrap-lines

  # 2) Forzar Docling sin OCR (PDFs digitales)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine docling --ocr-mode never --unwrap-lines

  # 3) Docling con OCR automático (si el texto queda corto, reintenta OCR full-page)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine docling --ocr spa --ocr-mode auto --unwrap-lines

  # 4) Docling con OCR forzado (todos los PDFs)
  python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --engine docling --ocr spa --ocr-mode force --unwrap-lines

Requerimientos:
  - markitdown[pdf] para usar MarkItDown con PDFs
  - docling para usar Docling (y opcionalmente Tesseract si vas a usar OCR)
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
from typing import Iterable, Tuple, Dict, Any, Optional

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
    """Devuelve PDFs de un archivo o recursivamente de una carpeta."""
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
        return
    if not path.exists():
        return
    for p in sorted(path.rglob("*.pdf")):
        yield p

def _maybe_set_tessdata_prefix():
    """En macOS, si no hay TESSDATA_PREFIX, probamos rutas de Homebrew para Tesseract."""
    if platform.system() != "Darwin":
        return
    if os.getenv("TESSDATA_PREFIX"):
        return
    for p in ("/opt/homebrew/share/tessdata", "/usr/local/share/tessdata"):
        if os.path.isdir(p):
            os.environ["TESSDATA_PREFIX"] = p
            break

def _tesseract_has_lang(lang: str) -> bool:
    """Verifica que Tesseract tenga instalado el idioma (p.ej., 'spa')."""
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
    """
    Une palabras cortadas por guion + salto de línea:
      "higie-\nne" → "higiene"
    Heurística: letra/numero antes y después del salto.
    """
    # Caso básico: "palabra-\ncontinuacion"
    s = re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])-\n([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])', r'\1\2', s)
    # Si hubiera espacios accidentales: "- \n" o "-\n "
    s = re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])-\s*\n\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9])', r'\1\2', s)
    return s

def unwrap_paragraphs_smart(s: str) -> str:
    """
    Une saltos de línea de párrafos (cortes por diseño) manteniendo
    saltos reales (fin de oración, títulos, listas, etc.).
    """
    def is_list_like(line: str) -> bool:
        return bool(re.match(r'^\s*(?:[-*•‣·]|[0-9]+[.)]|[A-Za-z]\))\s+', line))

    def is_heading(line: str) -> bool:
        t = line.strip()
        if not t:
            return False
        # Heurística: títulos cortos y con alta proporción de MAYÚSCULAS o Title Case breve
        if len(t) > 120:
            return False
        words = t.split()
        caps_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
        title_case = (len(words) <= 10) and all(w[:1].isupper() for w in words if w)
        # También considerar líneas con ## típicas de markdown
        md_heading = t.startswith("#")
        return caps_ratio > 0.6 or title_case or md_heading

    def ends_hard(line: str) -> bool:
        return bool(re.search(r'[.!?¿¡:;»”)\]]\s*$', line))

    def looks_wrapped(prev: str, cur: str) -> bool:
        # no unir si la actual es lista/título
        if is_list_like(cur) or is_heading(cur):
            return False
        # si la anterior cierra “fuerte”, probablemente es fin de oración
        if ends_hard(prev):
            return False
        # si la actual empieza en minúscula o con puntuación débil → unir
        if re.match(r'^\s*[,\.;:)\]]', cur):
            return True
        m = re.match(r'^\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])', cur)
        if m and m.group(1).islower():
            return True
        # si la anterior termina en palabra “conectora” → casi seguro corte por maquetado
        if re.search(
            r'\b(?:de|la|el|y|o|u|que|para|en|con|por|del|al|un|una|unos|unas|los|las|se|su|sus|lo|le|les|si|no|como|pero|más|menos)$',
            prev.strip(), re.IGNORECASE
        ):
            return True
        # heurística por longitud similar (líneas largas típicas de párrafo envuelto)
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
        # conservar líneas en blanco (separan párrafos)
        if not prev.strip() or not line.strip():
            out.append(line)
            continue
        if looks_wrapped(prev, line):
            out[-1] = prev.rstrip() + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)

def clean_text(
    s: str,
    unwrap_mode: str = "smart",
    unhyphenate: bool = True
) -> str:
    """
    Limpieza mínima + heurísticas de reflujo de texto.
    unwrap_mode: "off" | "simple" | "smart"
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    if unhyphenate:
        s = unhyphenate_linebreaks(s)

    if unwrap_mode == "simple":
        # unir líneas que no parecen fin de oración
        s = re.sub(r'(?<![.!?:;»”)\]])\n(?!\n)', ' ', s)
    elif unwrap_mode == "smart":
        s = unwrap_paragraphs_smart(s)

    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------- conversores ----------
def convert_with_markitdown(pdf_path: Path) -> str:
    """
    Convierte PDF → texto usando MarkItDown.
    Requiere: pip install "markitdown[pdf]"
    """
    try:
        from markitdown import MarkItDown  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MarkItDown no está instalado. Instalalo con: pip install 'markitdown[pdf]'"
        ) from e

    md = MarkItDown()
    try:
        res = md.convert(pdf_path)
        # Preferimos texto plano si está disponible; si no, markdown.
        text = getattr(res, "text_content", None) or getattr(res, "markdown", None) or str(res)
        return text
    except Exception as e:
        msg = str(e)
        if "MissingDependencyException" in msg:
            raise RuntimeError(
                "MarkItDown reconoce PDF pero faltan dependencias. Instalá: pip install 'markitdown[pdf]'"
            ) from e
        raise

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
    """
    Devuelve (texto, info). info = {used_ocr, fallback_triggered, engine, len}
    """
    try:
        # chequeo de import temprano
        import docling  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError("Docling no está instalado. Instalalo con: pip install docling") from e

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

    # auto: primero sin OCR
    conv_no = get_docling_converter_no_ocr()
    text_no = _run(conv_no)
    if len(text_no) >= fallback_min_chars:
        info.update({"used_ocr": False, "fallback_triggered": False, "engine": None, "len": len(text_no)})
        return text_no, info

    # Fallback con OCR full-page
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
    engine: str = "auto",
    # limpieza
    unwrap_mode: str = "smart",
    do_unwrap: bool = True,
    unhyphenate: bool = True,
    # docling params
    ocr_mode: str = "auto",
    ocr_engine: str = "tesseract-cli",
    ocr_lang: Optional[str] = None,
    fallback_min_chars: int = 300,
) -> Tuple[str, Dict[str, Any]]:
    """
    Convierte un PDF a texto con el motor indicado.
    engine: "auto" | "markitdown" | "docling"
    """
    if engine not in {"auto", "markitdown", "docling"}:
        raise ValueError("engine debe ser auto | markitdown | docling")

    def _post(text: str) -> str:
        return clean_text(
            text,
            unwrap_mode=(unwrap_mode if do_unwrap else "off"),
            unhyphenate=unhyphenate,
        )

    # preferimos MarkItDown
    if engine in {"auto", "markitdown"}:
        try:
            raw = convert_with_markitdown(pdf_path)
            return _post(raw), {"engine": "markitdown", "used_ocr": False, "len": len(raw)}
        except Exception as e:
            if engine == "markitdown":
                raise
            # en auto, si falla, caemos a docling
            print(f"[INFO] MarkItDown falló para {pdf_path.name}. Fallback a Docling. Motivo: {e}", file=sys.stderr)

    # Docling branch
    if engine in {"auto", "docling"}:
        # Preflight OCR (solo si corresponde)
        if ocr_mode in ("auto", "force") and ocr_lang:
            if ocr_engine == "tesseract-cli":
                _maybe_set_tessdata_prefix()
                if shutil.which("tesseract") is None:
                    print("[WARN] No encontré 'tesseract' en PATH. Instalalo (macOS: brew install tesseract) o cambiá a --ocr-engine ocrmac.", file=sys.stderr)
                else:
                    lang_chk = _normalize_ocr_lang(ocr_lang) or "spa"
                    if lang_chk != "auto" and not _tesseract_has_lang(lang_chk):
                        print(f"[WARN] Tesseract no tiene el idioma '{lang_chk}'. Instalalo (p.ej. 'spa'). Sigo con fallback…", file=sys.stderr)
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

    # no debería llegar
    raise RuntimeError("No se pudo convertir el PDF")

# ---------- CLI ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Convierte PDFs → TXT con MarkItDown (preferido) o Docling (con OCR).")
    ap.add_argument("-i", "--input", default=str(DEFAULT_IN_DIR), help=f"Archivo o carpeta de PDFs (default: {DEFAULT_IN_DIR})")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT_DIR), help=f"Carpeta de salida para .txt (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--engine", choices=["auto", "markitdown", "docling"], default="auto",
                    help="Motor de conversión (default: auto → intenta MarkItDown, fallback Docling).")

    # Limpieza/normalización
    ap.add_argument("--unwrap-lines", action="store_true", help="Desenrollar saltos de línea dentro de párrafos.")
    ap.add_argument("--unwrap-strategy", choices=["off", "simple", "smart"], default="smart",
                    help="Estrategia de desenrollado de líneas (default: smart).")
    ap.add_argument("--no-unhyphenate", action="store_true", help="No unir palabras cortadas con guion al final de línea.")

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
                ocr_mode=args.ocr_mode,
                ocr_engine=args.ocr_engine,
                ocr_lang=args.ocr,
                fallback_min_chars=args.min_chars_ocr,
            )
            out_txt.write_text(text, encoding="utf-8")

            post = ""
            if info.get("engine", "").startswith("docling"):
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
