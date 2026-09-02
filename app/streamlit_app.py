"""
Milestone 7 — Interface
=======================

Chat simples sobre as demonstrações financeiras do Flamengo.

    streamlit run app/streamlit_app.py

Mostra, além da resposta, OS TRECHOS que o modelo usou — para você conferir se
a resposta está ancorada no documento ou se o modelo inventou.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config          # noqa: E402
from src.query import MODES, guess_year  # noqa: E402
from src.rag import get_llm, run_stream  # noqa: E402

st.set_page_config(page_title="RAG · Demonstrações Flamengo", page_icon="📊", layout="centered")


@st.cache_resource(show_spinner="Preparando (embeddings + LLM)...")
def _warm(model: str):
    from src.embed_index import get_embed_model
    from src.rag import is_claude
    get_embed_model()
    if not is_claude(model):   # aquece o Ollama local; para o Claude seria chamada paga
        try:
            get_llm(model).complete("ok")
        except Exception:
            pass


def _available_models() -> list[str]:
    import os

    import httpx
    try:
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json()
        models = sorted(m["name"] for m in tags.get("models", []))
    except Exception:
        models = [config.OLLAMA_MODEL]
    if os.getenv("ANTHROPIC_API_KEY"):        # opção paga, só se houver chave no .env
        from src.rag import CLAUDE_MODELS
        models += list(CLAUDE_MODELS)
    return models


st.title("Demonstrações financeiras do Flamengo")
st.caption("RAG 100% local — busca nos PDFs de 2022–2026 e responde só com base neles.")

with st.sidebar:
    st.header("Configuração")
    mode = st.selectbox("Retrieval", MODES, index=MODES.index("bm25"),
                        help="bm25 = busca lexical (melhor aqui). hybrid = lexical + semântica.")
    k = st.slider("Trechos recuperados (k)", 1, 6, config.GENERATION_TOP_K,
                  help="Mais trechos = contexto maior = resposta mais lenta nesta CPU.")
    models = _available_models()
    model = st.selectbox("Modelo (Ollama)", models,
                         index=models.index(config.OLLAMA_MODEL) if config.OLLAMA_MODEL in models else 0)
    strategy = st.radio("Chunking", ["fixed", "structural"], horizontal=True)
    st.divider()
    st.caption("O ano é inferido da pergunta (ex: \"...em 2024?\"). Force abaixo se precisar.")
    year_override = st.text_input("Forçar exercício", placeholder="ex: 2024")

_warm(model)

question = st.text_input("Sua pergunta", placeholder="Qual foi a receita operacional líquida em 2023?")

if st.button("Perguntar", type="primary", disabled=not question):
    year = year_override.strip() or guess_year(question)

    with st.spinner("Buscando trechos..."):
        hits, tokens = run_stream(question, strategy=strategy, mode=mode, k=k,
                                  year=year or None, model=model)
    if not hits:
        st.warning("Nenhum trecho recuperado.")
        st.stop()

    st.markdown("### Resposta")
    box = st.empty()
    t0 = time.time()
    acc, last = "", 0.0
    try:
        with st.spinner("Gerando… o 1º token pode levar ~30 s (o modelo lê o contexto)."):
            for tok in tokens:
                acc += tok
                if time.time() - last > 0.3:        # throttle: texto puro durante o stream
                    box.text(acc)
                    last = time.time()
    except Exception as e:                          # Ollama caiu / timeout / modelo ocupado
        st.error(f"Falha na geração: {e}")
        st.stop()
    box.markdown(acc or "_(resposta vazia)_")       # render final com markdown
    st.caption(f"{time.time() - t0:.0f}s · {mode} · k={k} · {model}" + (f" · ano {year}" if year else ""))

    st.markdown("### Trechos usados (confira a fundamentação)")
    for i, h in enumerate(hits):
        m = h.node.metadata
        sc = f"{h.score:.3f}" if h.score is not None else "—"
        with st.expander(f"#{i + 1} · {m.get('source', '?')} · {m.get('header_path', '-')} · score {sc}"):
            st.text(h.node.get_content())
