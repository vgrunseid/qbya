Prospectos RAG Pipeline

Pipeline de Recuperación Aumentada con Generación (RAG) para prospectos médicos. Convierte PDFs a texto, detecta secciones, genera chunks con metadata, indexa embeddings en Chroma, y responde preguntas usando un LLM local (Ollama o llama.cpp) con contexto citado.

📦 Estructura del proyecto
.
├── 1-pdf_to_txt.py        # PDFs → TXT (Docling, opcional OCR Tesseract)
├── 2-transform.py         # TXT → chunks.jsonl (detección de secciones + metadata)
├── 3-embeddings.py        # chunks.jsonl → Chroma (embeddings OpenAI o E5 local)
├── 4-retrieve.py          # Recuperación (y respuesta con LLM local)
├── out_txt/               # TXT generados
├── out_chunks/            # chunks.jsonl
└── chroma/
    └── prospectos/        # base persistida de Chroma

🛠️ Requisitos

Python 3.10+

(Opcional) Tesseract para OCR:

macOS: brew install tesseract

Debian/Ubuntu: sudo apt-get install tesseract-ocr

(Opcional) Ollama para LLM local:

macOS: brew install ollama y luego ollama pull llama3.1

Dependencias Python:
pip install -r requirements.txt

Si usás OpenAI embeddings, definí OPENAI_API_KEY.

🚀 Quickstart (de punta a punta)
# 0) instalar dependencias
pip install -r requirements.txt

# 1) PDFs → TXT (usar --ocr es si hay escaneos)
python 1-pdf_to_txt.py -i pdfs_crudos -o out_txt --ocr es

# 2) TXT → chunks.jsonl (con secciones + prefijo [SECCIÓN: ...])
python 2-transform.py -i out_txt -o out_chunks/chunks.jsonl

# 3) Indexar en Chroma con E5 (local, sin API)
rm -rf chroma/prospectos
python 3-embeddings.py -f out_chunks/chunks.jsonl -p chroma/prospectos -c prospectos \
  --provider e5 --e5-model intfloat/multilingual-e5-base

# 4) Recuperar y responder con LLM local (Ollama)
ollama pull llama3.1
python 4-retrieve.py -q "contraindicaciones de ozempic" \
  -p chroma/prospectos -c prospectos \
  --provider e5 --e5-model intfloat/multilingual-e5-base \
  --auto --rerank keyword --answer --llm-backend ollama --llm-model llama3.1 -k 5

🗺️ Diagrama del flujo
flowchart LR
  A[PDFs] --> B[1-pdf_to_txt.py\nDocling (+OCR opcional)]
  B --> C[TXT limpios\nout_txt/]
  C --> D[2-transform.py\nSecciones + chunks + metadata]
  D --> E[chunks.jsonl\nout_chunks/]
  E --> F[3-embeddings.py\nEmbeddings (OpenAI o E5)]
  F --> G[Chroma persistido\nchroma/prospectos/]
  G --> H[4-retrieve.py\nsimilaridad + filtros + (rerank)]
  H --> I[LLM local\n(Ollama/llama.cpp)\nrespuesta con citas]

📘 Detalle por script
1) 1-pdf_to_txt.py — Convertir PDFs → TXT

Convierte PDFs a UTF-8 con Docling. Si pasás --ocr, usa Tesseract (ideal para escaneos). Limpia NBSP/BOM y normaliza saltos.

Uso:

# sin OCR
python 1-pdf_to_txt.py

# con OCR en español
python 1-pdf_to_txt.py --ocr es

# rutas personalizadas + OCR full page
python 1-pdf_to_txt.py -i ./pdfs_crudos -o ./out_txt --ocr es --force-ocr


Parámetros clave:

-i/--input (default: ./pdfs_crudos)

-o/--output (default: ./out_txt)

--ocr es|eng|...

--force-ocr

--skip-existing

Salida: un .txt por PDF en out_txt/.

2) 2-transform.py — TXT → chunks.jsonl (secciones + metadata)

Detecta secciones (p. ej., POSOLOGÍA, CONTRAINDICACIONES), les asigna un nombre canónico y chunkifica el contenido.
Si no encuentra encabezado confiable, aplica fallback por keywords en el cuerpo (p. ej., “posología/dosis/modo de administración”, etc.).
Inserta un prefijo "[SECCIÓN: ...]" al texto del chunk para mejorar la recuperación.

Uso:

# defaults: in=./out_txt, out=./out_chunks/chunks.jsonl
python 2-transform.py

# tamaños de chunk y solape
python 2-transform.py -i out_txt -o out_chunks/chunks.jsonl --chunk-size 1200 --chunk-overlap 200

# desactivar prefijo y fallback por cuerpo
python 2-transform.py --no-prefix-section --no-body-fallback

# fallback más sensible (1 keyword basta)
python 2-transform.py --fallback-min-hits 1


Metadata por chunk:

drug_name (del nombre del archivo .txt, p. ej. ibupirac flex)

drug_root (primera palabra, p. ej. ibupirac)

doc_name (ej. ozempic.txt)

section_raw, section_canonical (o UNKNOWN)

section_match_score, section_inferred_from_body, section_infer_score

section_start_line, chunk_index_in_section

Salida: out_chunks/chunks.jsonl (una línea JSON por chunk).

3) 3-embeddings.py — Indexar en Chroma

Lee el chunks.jsonl, calcula embeddings con OpenAI o E5 local y guarda todo en una colección Chroma persistida.

Uso (E5 local):

python 3-embeddings.py -f out_chunks/chunks.jsonl -p chroma/prospectos -c prospectos \
  --provider e5 --e5-model intfloat/multilingual-e5-base


Uso (OpenAI):

export OPENAI_API_KEY=...
python 3-embeddings.py -f out_chunks/chunks.jsonl -p chroma/prospectos -c prospectos \
  --provider openai --openai-model text-embedding-3-small


Opciones útiles:

--batch-size 256

--skip-existing (no reindexa IDs ya presentes)

-p y -c para manejar varias bases o colecciones

Salida: base en chroma/prospectos/ con vectores + metadata.

Importante: consultá usando el mismo provider/modelo con el que indexaste (E5 ↔ E5, OpenAI ↔ OpenAI).

4) 4-retrieve.py — Recuperación (+ respuesta con LLM)

Recupera top-K fragmentos por similaridad (con filtros de metadata compatibles con tu Chroma), hace reranking para priorizar la sección pedida y, opcionalmente, usa un LLM local para sintetizar respuesta con citas [n].

Uso (solo recuperar):

python 4-retrieve.py -q "contraindicaciones de ozempic" \
  -p chroma/prospectos -c prospectos \
  --provider e5 --e5-model intfloat/multilingual-e5-base -k 5


Uso (recuperar + responder con Ollama):

ollama pull llama3.1
python 4-retrieve.py -q "posología de sertal" \
  -p chroma/prospectos -c prospectos \
  --provider e5 --e5-model intfloat/multilingual-e5-base \
  --auto --rerank keyword --answer --llm-backend ollama --llm-model llama3.1 -k 5


Parámetros clave:

Recuperación: -q, -p, -c, --provider, --e5-model/--openai-model, -k

Filtros exactos: --drug (minúsculas), --section (MAYÚSCULAS)

Autodetección desde la pregunta: --auto

Rerank: --rerank off|keyword|cross (para cross instalar sentence-transformers)

LLM local: --answer, --llm-backend ollama|llamacpp, --llm-model o --llm-model-path (.gguf), --llm-n-ctx, --llm-temperature, etc.

Salida: ranking de fragmentos y, si --answer, respuesta con citas + listado de fuentes.

🔍 Consejos de calidad

Secciones en embeddings: el prefijo "[SECCIÓN: ...]" dentro del chunk mejora notablemente la relevancia.

Filtros de Chroma: tu versión soporta "$eq", "$in", "$ne", "$gt", "$gte", "$lt", "$lte".
No uses "$contains". Para “contiene”, traé un lote y filtrá en Python.

Secciones UNKNOWN: si faltan canónicas (ej. POSOLOGÍA en Sertal), ajustá SECTION_HINTS o baja --fallback-min-hits a 1 y reindexá.

Nombres de drogas: drug_name proviene del nombre del archivo (ibupirac flex, sertal gotas). Para agrupar por “familia”, usá drug_root.

🧪 Verificación opcional (debug)

Si querés listar qué drogas y secciones hay en la colección (útil para elegir filtros):

# drogas disponibles
from langchain_chroma import Chroma
emb = type("E", (), {"embed_documents": lambda *a, **k: [], "embed_query": lambda *a, **k: []})()
vs = Chroma(collection_name="prospectos", persist_directory="chroma/prospectos", embedding_function=emb)
res = vs._collection.get(include=["metadatas"], limit=100000).get("metadatas", [])
print(sorted({(m or {}).get("drug_name","") for m in res if m}))

# secciones para 'ozempic'
from collections import Counter
res = vs._collection.get(where={"drug_name":{"$eq":"ozempic"}}, include=["metadatas"], limit=100000)
cnt = Counter((m or {}).get("section_canonical") or (m or {}).get("section_raw") or "UNKNOWN" for m in res.get("metadatas",[]))
print(cnt.most_common())

🧯 Troubleshooting

No devuelve resultados:

Confirmá que chroma/prospectos existe y count > 0.

Usá el mismo embedding model que indexaste.

Probá sin filtros; luego agregalos de a uno.

Verificá cómo se llama exactamente drug_name (sale del filename).

Filtros rompen:

Usá solo "$eq", "$in", "$and", "$or".

Si querés “contiene”, filtrá en Python tras traer un lote grande.

Secciones mal detectadas:

Ajustá SECTION_HINTS en 2-transform.py o bajá --fallback-min-hits.

Re-ejecutá 2-transform.py y reindexá (rm -rf chroma/prospectos; python 3-embeddings.py ...).

Ollama no responde:

ollama serve corriendo y modelo descargado (ollama pull llama3.1).

Si usás llama.cpp, asegurate de pasar --llm-model-path al .gguf.# qbya
