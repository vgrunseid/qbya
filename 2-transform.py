#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-transform.py — Segmenta prospectos en SECCIONES y CHUNKS para RAG,
y obtiene el nombre comercial consultando a un LLM local (Ollama).

Novedades:
- drug_name se obtiene pidiéndoselo al LLM con el PDF (o TXT) como contexto.
- Eliminado todo lo previo de heurísticas/regex para detectar la marca.

Ejemplos:
  python 2-transform.py -i out_txt -o out_chunks/chunks.jsonl \
      --pdf-root pdfs_crudos --brand-llm-model llama3.1 --max-files 10
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

# ==========================
# Sección: utilidades texto
# ==========================

def ustrip(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).strip()

def is_likely_heading(line: str) -> bool:
    s = ustrip(line)
    if not s:
        return False
    if len(s) <= 80:
        if s == s.upper() and re.search(r"[A-ZÁÉÍÓÚÜ]", s):
            return True
        if s.endswith(":") and not re.search(r"[\.!?]$", s[:-1]):
            return True
        if s.startswith(("#", "##")) and len(s.split()) <= 10:
            return True
        if s[0].isupper() and not s.endswith(".") and len(s.split()) <= 10:
            return True
    return False

def split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n{2,}", text)
    paras = [ustrip(p) for p in paras if ustrip(p)]
    return paras

def sentence_split(s: str) -> List[str]:
    # Respetar listas con bullets/guiones
    if "\n" in s and len(s) < 400:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if all(len(ln) <= 200 for ln in lines) and any(ln.startswith(("-", "•", "*")) for ln in lines):
            return lines
    # Final de oración; cortar piezas demasiado largas por ;:
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÜ0-9\-•*])", s.strip())
    out: List[str] = []
    for p in parts:
        pp = p.strip()
        if not pp:
            continue
        if len(pp) > 800:
            sub = re.split(r"(?<=[;:])\s+", pp)
            out.extend([x.strip() for x in sub if x.strip()])
        else:
            out.append(pp)
    return out

def approx_token_count(s: str) -> int:
    # Conteo aprox. por "palabras/signos"
    return len(re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE))

# =======================================
# Sección: mapeo y prototipos de secciones
# =======================================

SECTION_ALIASES_PATTERNS: Dict[str, List[str]] = {
    "DENOMINACIÓN / COMPOSICIÓN": [
        r"\b(denominaci[oó]n|composici[oó]n|f[oó]rmula|ingredientes?)\b",
    ],
    "ACCIÓN TERAPÉUTICA": [
        r"\b(acci[oó]n\s+terap[eé]utica|grupo\s+farmacoterap[eé]utico|clasificaci[oó]n\s+atc)\b",
    ],
    "ACCIÓN FARMACOLÓGICA": [
        r"\b(acci[oó]n\s+farmacol[oó]gica|propiedades?\s+farmacol[oó]gic[as]|farmacodinamia)\b",
    ],
    "FARMACOCINÉTICA": [
        r"\b(farmacocin[eé]tica|absorci[oó]n|distribuci[oó]n|metaboli[sz]aci[oó]n|eliminaci[oó]n|vida\s+media|t\s*1/2)\b",
    ],
    "INDICACIONES": [
        r"\b(indicaciones?|usos?\s+terap[eé]uticos|est[aá]\s+indicado)\b",
    ],
    "POSOLOGÍA Y MODO DE ADMINISTRACIÓN": [
        r"\b(posolog[ií]a|dosis|d[oó]sis|modo\s+de\s+(uso|administraci[oó]n)|v[ií]a\s+de\s+administraci[oó]n)\b",
    ],
    "CONTRAINDICACIONES": [
        r"\b(contraindicaciones?|no\s+(utilizar|usar)(\s+en)?)\b",
    ],
    "ADVERTENCIAS Y PRECAUCIONES": [
        r"\b(advertencias?|precauciones?|precauci[oó]n(es)?\s+de\s+uso)\b",
    ],
    "INTERACCIONES": [
        r"\b(interacciones?|interacci[oó]n\s+con\s+otras\s+(drogas|sustancias|medicamentos))\b",
    ],
    "REACCIONES ADVERSAS": [
        r"\b(reacciones?\s+adversas?|eventos?\s+adversos?|efectos?\s+adversos?)\b",
    ],
    "SOBREDOSIS": [
        r"\b(sobredosis|intoxicaci[oó]n|tratamiento\s+de\s+la\s+sobredosis)\b",
    ],
    "EMBARAZO Y LACTANCIA": [
        r"\b(embarazo|lactancia|fertilidad|uso\s+en\s+embarazo|madres?\s+lactantes)\b",
    ],
    "CONSERVACIÓN / ALMACENAMIENTO": [
        r"\b(conservaci[oó]n|almacenamiento|condiciones?\s+de\s+conservaci[oó]n|vencimiento|vence)\b",
    ],
    "PRESENTACIÓN": [
        r"\b(presentaci[oó]n|envase|forma\s+farmac[eé]utica)\b",
    ],
}

def compile_aliases() -> Dict[str, List[re.Pattern]]:
    comp: Dict[str, List[re.Pattern]] = {}
    for canonical, pats in SECTION_ALIASES_PATTERNS.items():
        comp[canonical] = [re.compile(p, re.IGNORECASE) for p in pats]
    return comp

ALIASES_RX = compile_aliases()

def match_canonical(title: str) -> Optional[Tuple[str, str]]:
    t = ustrip(title)
    if not t:
        return None
    for canonical, rx_list in ALIASES_RX.items():
        for rx in rx_list:
            if rx.search(t):
                return canonical, t
    t_up = t.upper().replace("  ", " ")
    for canonical in ALIASES_RX.keys():
        if canonical in t_up or t_up in canonical:
            return canonical, t
    return None

# =========================
# Sección: estructuras datos
# =========================

@dataclass
class Section:
    canonical: Optional[str]
    raw_title: Optional[str]
    start_char: int
    end_char: int
    text: str
    confidence: float = 1.0

# ======================================
# Sección: clasificador semántico (opcional)
# ======================================

SECTION_PROTOTYPES: Dict[str, List[str]] = {
    "ACCIÓN FARMACOLÓGICA": [
        "Mecanismo de acción y propiedades farmacológicas del fármaco.",
        "Describe cómo actúa el medicamento a nivel farmacodinámico.",
        "Efectos sobre receptores, vías y fisiología."
    ],
    "ACCIÓN TERAPÉUTICA": [
        "Grupo farmacoterapéutico y clasificación ATC.",
        "Para qué se utiliza en términos terapéuticos generales."
    ],
    "POSOLOGÍA Y MODO DE ADMINISTRACIÓN": [
        "Dosis recomendada, forma y frecuencia de administración.",
        "Ajustes de dosis en poblaciones especiales."
    ],
    "CONTRAINDICACIONES": [
        "Situaciones en las que no debe utilizarse el medicamento.",
        "Hipersensibilidad al principio activo o a excipientes."
    ],
    "ADVERTENCIAS Y PRECAUCIONES": [
        "Advertencias especiales y precauciones de empleo.",
        "Riesgos, monitorización y uso cuidadoso."
    ],
    "REACCIONES ADVERSAS": [
        "Efectos adversos reportados y su frecuencia.",
        "Perfil de seguridad y tolerabilidad."
    ],
    "INTERACCIONES": [
        "Interacciones con otros medicamentos o alimentos.",
        "Cambios de efecto por administración concomitante."
    ],
    "FARMACOCINÉTICA": [
        "Absorción, distribución, metabolismo y eliminación.",
        "Parámetros como Cmax, AUC y vida media."
    ],
    "EMBARAZO Y LACTANCIA": [
        "Uso durante el embarazo, lactancia y fertilidad.",
        "Riesgos para el feto o el lactante."
    ],
    "CONSERVACIÓN / ALMACENAMIENTO": [
        "Condiciones de almacenamiento y caducidad.",
        "Temperatura, luz y otras condiciones de conservación."
    ],
    "DENOMINACIÓN / COMPOSICIÓN": [
        "Principio activo, excipientes y denominación del producto.",
        "Concentraciones y forma farmacéutica."
    ],
    "INDICACIONES": [
        "Indicaciones terapéuticas autorizadas.",
        "Cuándo está indicado el medicamento."
    ],
    "PRESENTACIÓN": [
        "Presentaciones disponibles y formas de envase."
    ],
}

class SemanticLabeler:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", threshold: float = 0.33):
        from sentence_transformers import SentenceTransformer
        self.np = __import__("numpy")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.threshold = float(threshold)
        self.proto_texts: List[Tuple[str, str]] = []
        for canonical, lst in SECTION_PROTOTYPES.items():
            for t in lst:
                self.proto_texts.append((canonical, f"passage: {t}"))
        self.proto_emb = self.model.encode([t for _, t in self.proto_texts], normalize_embeddings=True)

    def label(self, text: str) -> Tuple[Optional[str], float]:
        if not text or not text.strip():
            return None, 0.0
        q = f"query: {text.strip()[:1500]}"
        emb = self.model.encode([q], normalize_embeddings=True)[0]
        sims_vec = self.proto_emb @ emb
        sims = float(self.np.max(sims_vec))
        idx = int(self.np.argmax(sims_vec))
        canonical = self.proto_texts[idx][0]
        if sims >= self.threshold:
            return canonical, float(sims)
        return None, float(sims)

# ==========================
# Sección: seccionado y chunking
# ==========================

def build_sections(text: str, use_semantic: bool = False, sem_labeler: Optional[SemanticLabeler] = None) -> List[Section]:
    paragraphs = split_paragraphs(text)
    sections: List[Section] = []

    cur_title_raw: Optional[str] = None
    cur_canonical: Optional[str] = None
    cur_buf: List[str] = []
    cur_start = 0

    offset = 0
    for para in paragraphs:
        pos = text.find(para, offset)
        if pos >= 0:
            offset = pos + len(para)

        looks_heading = is_likely_heading(para)
        m = match_canonical(para) if looks_heading else None

        if m:
            if cur_buf:
                sec_text = "\n\n".join(cur_buf).strip()
                end_char = offset - len(para) - 2 if pos >= 0 else cur_start + len(sec_text)
                sections.append(Section(
                    canonical=cur_canonical or None,
                    raw_title=cur_title_raw,
                    start_char=cur_start,
                    end_char=max(cur_start, end_char),
                    text=sec_text,
                    confidence=1.0 if cur_canonical else 0.0,
                ))
                cur_buf = []

            cur_title_raw = m[1]
            cur_canonical = m[0]
            cur_start = offset
            continue

        cur_buf.append(para)

    if cur_buf:
        sec_text = "\n\n".join(cur_buf).strip()
        end_char = len(text)
        sections.append(Section(
            canonical=cur_canonical or None,
            raw_title=cur_title_raw,
            start_char=cur_start,
            end_char=end_char,
            text=sec_text,
            confidence=1.0 if cur_canonical else 0.0,
        ))

    if use_semantic and sem_labeler is not None:
        for s in sections:
            if not s.canonical:
                can, score = sem_labeler.label(s.text[:2000])
                if can:
                    s.canonical = can
                    s.confidence = score
                else:
                    s.canonical = None
                    s.confidence = score

    return sections

def chunk_section_text(text: str, max_tokens: int = 500, overlap: int = 80) -> List[str]:
    if not text.strip():
        return []
    units = sentence_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for u in units:
        u_tok = approx_token_count(u)
        if cur and (cur_tok + u_tok) > max_tokens:
            chunk_text = " ".join(cur).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if overlap > 0 and chunks:
                tail = chunks[-1].split()
                keep = tail[-min(len(tail), overlap):]
                cur = [" ".join(keep)]
                cur_tok = approx_token_count(cur[0])
            else:
                cur = []
                cur_tok = 0
        cur.append(u)
        cur_tok += u_tok

    last = " ".join(cur).strip()
    if last:
        chunks.append(last)
    return chunks

# ==============================
# Sección: LLM para brand (Ollama)
# ==============================

def _pdf_to_text_with_markitdown(pdf_path: Path, max_chars: int = 6000) -> Optional[str]:
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        r = md.convert(str(pdf_path))
        text = getattr(r, "text_content", None) or str(r)
        text = ustrip(text)
        return text[:max_chars] if text else None
    except Exception:
        return None

def _pdf_to_text_with_pypdf(pdf_path: Path, max_pages: int = 3, max_chars: int = 6000) -> Optional[str]:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for i in range(min(len(reader.pages), max_pages)):
            pages.append(reader.pages[i].extract_text() or "")
        text = ustrip("\n".join(pages))
        return text[:max_chars] if text else None
    except Exception:
        return None

def extract_context_for_brand(txt_path: Path, pdf_candidate: Optional[Path], max_chars: int = 6000) -> str:
    """
    Preferimos PDF → texto (MarkItDown o PyPDF). Si no, usamos el TXT.
    Solo tomamos un recorte (cabecera + algo) para el prompt.
    """
    if pdf_candidate and pdf_candidate.exists():
        txt = _pdf_to_text_with_markitdown(pdf_candidate, max_chars=max_chars)
        if not txt:
            txt = _pdf_to_text_with_pypdf(pdf_candidate, max_chars=max_chars)
        if txt:
            return txt
    # fallback: usar TXT
    try:
        raw = txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = txt_path.read_text(encoding="latin-1", errors="ignore")
    return ustrip(raw)[:max_chars]

def query_ollama_brand(context: str, model: str = "llama3.1", host: str = "http://localhost:11434", timeout: float = 30.0) -> Optional[str]:
    """
    Pide SOLO el nombre comercial al LLM local (Ollama /api/chat).
    Devuelve una cadena corta (sin comillas ni símbolos) o None si falla.
    """
    import requests

    system_msg = (
        "Eres un extractor estricto. Devuelve únicamente el NOMBRE COMERCIAL "
        "del medicamento tal como figura en el prospecto, sin forma farmacéutica, "
        "sin dosis, sin principio activo, sin comillas ni texto adicional. "
        "Ejemplos de salida válida: 'ALPLAX XR', 'AMBISOME', 'APRIL'. "
        "Si hay variantes, devuelve la marca principal."
        "No anexes el nombre del laboratorio al nombre comercial"
    )
    user_msg = (
        "Contexto de prospecto entre llaves:\n"
        "{{\n" + context + "\n}}\n\n"
        "Salida (solo el nombre comercial, una sola línea):"
    )

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        brand = sanitize_brand_output(content)
        return brand or None
    except Exception:
        return None

def sanitize_brand_output(s: str) -> str:
    """
    Limpia la respuesta del LLM:
    - primera línea
    - quita comillas/símbolos/etiquetas
    - quita unidades/dosis si vinieran coladas
    - normaliza espacios
    """
    if not s:
        return ""
    line = s.strip().splitlines()[0]
    line = re.sub(r"^[\"'“”‘’\-\*\s]+|[\"'“”‘’\-\*\s]+$", "", line)
    # quitar marcas registradas y cosas entre paréntesis
    line = re.sub(r"\s*(?:®|™|TM)\b", "", line, flags=re.I)
    line = re.sub(r"\(.*?\)", "", line)
    # quitar dosis/formas típicas: mg, mcg, %, ml, comprimidos, solución, cápsulas, etc.
    line = re.sub(r"\b(\d+(?:[.,]\d+)?\s*(?:mg|mcg|μg|g|kg|ml|mL|%))\b", "", line, flags=re.I)
    line = re.sub(r"\b(comprimidos?|tabletas?|c[aá]psulas?|soluci[oó]n|suspensi[oó]n|jarabe|inyecci[oó]n|gel|crema)\b", "", line, flags=re.I)
    # múltiples espacios
    line = re.sub(r"\s{2,}", " ", line).strip(" -–—·:;.,")
    return line

def find_pdf_for(txt_file: Path, pdf_root: Optional[Path]) -> Optional[Path]:
    """
    Busca un PDF con stem similar al del TXT:
    - mismo directorio (mismo stem .pdf)
    - en pdf_root por stem exacto
    - en pdf_root por prefijo que contenga el stem (heurística simple)
    """
    stem = txt_file.stem.lower()
    # 1) mismo dir
    cand = txt_file.with_suffix(".pdf")
    if cand.exists():
        return cand
    # 2) pdf_root exacto
    if pdf_root and pdf_root.exists():
        exact = list(pdf_root.rglob(stem + ".pdf"))
        if exact:
            return exact[0]
        # 3) por contención de stem
        for p in pdf_root.rglob("*.pdf"):
            if stem in p.stem.lower() or p.stem.lower() in stem:
                return p
    return None

# ================
# Sección: IO
# ================

def read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1", errors="ignore")

def iter_input_files(p: Path) -> Iterable[Path]:
    if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in {".txt", ".md"}:
                yield f

# ================
# Sección: MAIN
# ================

def main() -> int:
    ap = argparse.ArgumentParser(description="Transforma .txt de prospectos en JSONL de chunks con secciones y brand por LLM (Ollama).")
    ap.add_argument("-i", "--input", required=True, help="Archivo o carpeta con .txt / .md")
    ap.add_argument("-o", "--output", required=True, help="Ruta de salida .jsonl")
    ap.add_argument("--pdf-root", help="Carpeta donde buscar los PDF originales para dar contexto al LLM")
    ap.add_argument("--max-files", type=int, default=0, help="Procesar como máximo N archivos (0 = sin límite)")

    # chunking
    ap.add_argument("--max-tokens", type=int, default=500, help="Tamaño aprox. de chunk (por palabras/tokens)")
    ap.add_argument("--overlap", type=int, default=80, help="Solape aprox. entre chunks (por palabras/tokens)")

    # etiquetas semánticas opcionales
    ap.add_argument("--semantic-labels", action="store_true", help="Clasifica secciones UNKNOWN con E5")
    ap.add_argument("--e5-model", default="intfloat/multilingual-e5-base", help="Modelo E5 para semántica")
    ap.add_argument("--sem-threshold", type=float, default=0.33, help="Umbral de similitud para rotular sección")

    # LLM (Ollama)
    ap.add_argument("--brand-llm-model", default="llama3.1", help="Modelo de Ollama para extraer nombre comercial")
    ap.add_argument("--brand-llm-host", default="http://localhost:11434", help="Host de Ollama (default http://localhost:11434)")
    ap.add_argument("--brand-context-chars", type=int, default=6000, help="Máx. caracteres de contexto que se envían al LLM")
    ap.add_argument("--brand-timeout", type=float, default=30.0, help="Timeout en segundos para la llamada a Ollama")

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_root = Path(args.pdf_root).expanduser().resolve() if args.pdf_root else None

    # semántica (opcional)
    sem_labeler = None
    if args.semantic_labels:
        try:
            sem_labeler = SemanticLabeler(model_name=args.e5_model, threshold=args.sem_threshold)
        except Exception as e:
            print(f"[WARN] No se pudo inicializar SemanticLabeler ({e}). Sigo sin semántica.", file=sys.stderr)
            sem_labeler = None

    total_chunks = 0
    total_files = 0
    processed_files = 0
    max_files = int(args.max_files) if args.max_files else 0

    with out_path.open("w", encoding="utf-8") as jf:
        for f in iter_input_files(in_path):
            if max_files and processed_files >= max_files:
                break

            total_files += 1
            raw = read_text(f)
            if not raw.strip():
                if args.verbose:
                    print(f"↷ {f.name}: vacío, salto")
                continue

            # Buscar PDF candidato para darle contexto al LLM
            pdf_path = find_pdf_for(f, pdf_root)
            context = extract_context_for_brand(f, pdf_path, max_chars=args.brand_context_chars)

            # Pedir nombre comercial al LLM (con fallback al stem si falla)
            brand = query_ollama_brand(
                context=context,
                model=args.brand_llm_model,
                host=args.brand_llm_host,
                timeout=args.brand_timeout,
            ) or f.stem  # fallback mínimo

            # construir secciones y chunks desde el TXT
            sections = build_sections(raw, use_semantic=bool(sem_labeler), sem_labeler=sem_labeler)

            if args.verbose:
                secs_ok = sum(1 for s in sections if s.canonical)
                secs_unk = len(sections) - secs_ok
                src = f"(PDF: {pdf_path.name})" if pdf_path else "(TXT)"
                print(f"→ {f.name}: secciones={len(sections)} (etiquetadas={secs_ok}, unknown={secs_unk})  drug={brand}  fuente={src}")

            doc_id = f.stem
            chunk_idx = 0
            for sec in sections:
                canonical = sec.canonical or "UNKNOWN"
                parts = chunk_section_text(sec.text, max_tokens=args.max_tokens, overlap=args.overlap)
                for part in parts:
                    rid = f"{doc_id}#{total_chunks:04d}"
                    rec = {
                        "id": rid,
                        "text": part,
                        "metadata": {
                            "doc_id": doc_id,
                            "doc_name": f.name,
                            "drug_name": brand.lower().strip(),
                            "section_canonical": canonical,
                            "section_raw": sec.raw_title,
                            "section_confidence": float(sec.confidence),
                            "chunk_index": chunk_idx,
                            "char_start": int(sec.start_char),
                            "char_end": int(sec.end_char),
                            "source_path": str(f),
                            "pdf_source": str(pdf_path) if pdf_path else None,
                        },
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunk_idx += 1
                    total_chunks += 1

            processed_files += 1

    print(f"Listo. Archivos procesados: {processed_files}  (vistos: {total_files})")
    print(f"Se escribieron {total_chunks} chunks → {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
