# tech-notes-rag

Pipeline RAG local para perguntas sobre as **demonstrações financeiras do
Clube de Regatas do Flamengo** (exercícios de 2022 a 2025 + Relatório de
Transparência do 2º Tri 2026). **Custo de nuvem: US$ 0** — OCR, embeddings,
banco vetorial e LLM rodam na própria máquina.

> Em construção. Progresso por milestone abaixo.
>
> Os PDFs de 2024 e 2025 não têm camada de texto (2024 = imagem por página,
> 2025 = texto vetorizado). São processados por OCR local (Tesseract + `por`).

## Stack

| Camada | Ferramenta |
|---|---|
| Framework RAG | LlamaIndex |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (ONNX/fastembed, roda em CPU) |
| Vector DB | ChromaDB (distância de cosseno) |
| LLM de geração | Ollama (Qwen2.5 / Llama 3.1) |
| Avaliação | RAGAS |
| Interface | Streamlit |

## Milestones

- [x] **1 — Ingestão + chunking** (`src/ingest.py`, `src/ocr.py`, `src/clean.py`)
- [x] **2 — Embeddings + indexação** (`src/embed_index.py`)
- [x] **3 — Retrieval** (`src/query.py`, sem LLM) — vector / bm25 / hybrid + filtro por ano
- [ ] 4 — Geração com LLM local
- [ ] 5 — Avaliação com RAGAS (`src/evaluate.py`)
- [ ] 6 — Experimento: chunking A vs. B, com/sem reranking
- [ ] 7 — Interface Streamlit + resultados no README

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# OCR (para os PDFs de 2024/2025):
winget install UB-Mannheim.TesseractOCR
curl -L -o models/tessdata/por.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/por.traineddata
```

## Uso — Milestone 1

```bash
# pipeline completo: extrai (ou OCR) -> limpa -> chunk
python -m src.ingest --source data/raw_docs --strategy structural --show 5
python -m src.ingest --source data/raw_docs --strategy fixed

# ver o efeito de cada regra de limpeza (antes/depois)
python -m src.clean
```

Fluxo: `data/raw_docs/*.pdf` → `data/extracted/*.md` (cache da extração/OCR)
→ `data/clean/*.md` (pós-limpeza) → `data/chunks/<estrategia>.jsonl`.

Flags úteis: `--no-clean` (pula a limpeza), `--source data/sample_docs` (roda
no excerto versionado, sem os PDFs grandes).

## Uso — Milestone 2

```bash
# indexa os chunks no ChromaDB (uma coleção por estratégia)
python -m src.embed_index --strategy structural --rebuild
python -m src.embed_index --strategy fixed --rebuild

# busca de teste (sem LLM ainda)
python -m src.embed_index --strategy structural --probe "saldo de caixa em 2023?"
```

Índice em `chroma_db/`. ~0.4 s/chunk nesta máquina (CPU).

## Uso — Milestone 3 (retrieval, sem LLM)

```bash
# compara busca densa / lexical / híbrida na mesma pergunta
python -m src.query "qual a provisão para contingências em 2024?" --compare

# um modo só, com filtro por exercício
python -m src.query "provisão para contingências" --mode bm25 --year 2024 --k 4
```

`--mode vector|bm25|hybrid` · `--year 2024` (filtra `doc_year`) · `--k N` ·
`--strategy fixed|structural`.
