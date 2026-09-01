"""
Configuração central do projeto.

Tudo que é "parâmetro de decisão" do pipeline mora aqui, para que os
experimentos (mudar chunk_size, trocar de modelo, etc.) sejam feitos num
lugar só e fiquem versionados.
"""

from pathlib import Path

# --- Caminhos ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"        # seus documentos reais (não versionado)
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"  # exemplos versionados, para rodar de cara
EXTRACTED_DIR = DATA_DIR / "extracted"      # cache: PDF -> Markdown (não versionado)
CHUNKS_DIR = DATA_DIR / "chunks"            # saída do ingest.py (não versionado)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"     # índice persistido (milestone 2)
TESSDATA_DIR = PROJECT_ROOT / "models" / "tessdata"  # por.traineddata (não versionado)

# Pastas cujo nome começa com este prefixo são ignoradas na ingestão
# (ex: data/raw_docs/_ocr_pendente/ com PDFs escaneados aguardando OCR).
IGNORE_DIR_PREFIX = "_"

# --- Chunking (Milestone 1) -------------------------------------------------
# Estratégia "fixed": tamanho fixo em tokens, com sobreposição entre vizinhos.
CHUNK_SIZE = 512       # tokens por chunk
CHUNK_OVERLAP = 64     # tokens repetidos entre chunks consecutivos (~12%)

# Estratégia "structural": corta pelos títulos do Markdown. Seções muito
# grandes são reparticionadas em blocos deste tamanho para não estourar o
# contexto lá na geração.
# Nota: demonstrativos financeiros têm tabelas grandes (um balanço patrimonial
# inteiro pode passar de 800 tokens). Mantemos o teto alto para não cortar uma
# tabela no meio — se cortar, o retrieval traz meia tabela e o número some.
STRUCTURAL_MAX_TOKENS = 1200

# Tokenizer usado para CONTAR tokens (aproximação; o tokenizer real do
# modelo de embedding difere, mas serve para comparar estratégias).
TOKEN_ENCODING = "cl100k_base"

# Chunks menores que isto são fundidos ao vizinho (título solto, linha órfã).
MIN_CHUNK_TOKENS = 24

# --- Embeddings (Milestone 2) --------------------------------------------
# Roda em CPU via ONNX (fastembed). BGE-M3 seria melhor em qualidade, mas ~7s
# por chunk/pergunta nesta máquina — inviável para uso interativo. MiniLM
# multilíngue: ~0.3s/chunk, PT ok. Ver notes/ e memory/ para o histórico.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32         # chunks por lote na vetorização

def collection_name(strategy: str) -> str:
    """Uma coleção do Chroma por estratégia de chunking, para comparar as duas."""
    return f"chunks_{strategy}"

# --- LLM local (Milestone 4) --------------------------------------------
# Avaliação (docs/avaliacao.md): llama3.2:3b 65% acerto / ~64s; qwen2.5:7b
# 71% / ~153s (mantém abstenção); gemma2:2b 59% e quebra abstenção.
# 3b é o padrão pelo equilíbrio. Troque no seletor da interface se quiser o 7b.
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 900          # segundos. Prefill de prompt longo num i5-8265U
                             # é MUITO lento (~10-15 tok/s) -> respostas de minutos.
OLLAMA_CONTEXT_WINDOW = 4096  # system + k chunks + pergunta. Menor = prefill mais rápido.
OLLAMA_KEEP_ALIVE = "30m"     # mantém o modelo na RAM entre perguntas
OLLAMA_NUM_PREDICT = 256      # teto de tokens gerados. Sem isto o modelo às vezes
                             # divaga por 1-2 min. A resposta é curta (1-3 frases).
GENERATION_TEMPERATURE = 0.1  # baixa = factual e reproduzível

# LLM juiz do RAGAS (Milestone 5). Local por padrão = grátis mas impreciso.
# Troque por um modelo forte para números confiáveis.
RAGAS_JUDGE_MODEL = "llama3.2:3b"

# Caractere-limite por chunk no prompt. Cortar o contexto acelera o prefill,
# mas a avaliação mostrou que cortar demais derruba a acurácia (perde o final
# de tabelas). 1100 é o valor que segurou os 65%.
CONTEXT_CHARS_PER_CHUNK = 1100

# --- Retrieval (Milestone 3) -------------------------------------------
TOP_K = 5

# k usado na GERAÇÃO (Milestone 4). k=2 é ~30% mais rápido, mas custou -24
# pontos de acurácia na avaliação (nota explicativa despencou de 60% p/ 20%).
# k=3 é o mínimo para a resposta certa aparecer no contexto.
GENERATION_TOP_K = 3
