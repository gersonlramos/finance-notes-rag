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
from src.query import MODES, guess_year, retrieve  # noqa: E402
from src.rag import SYSTEM_PROMPT, build_context, build_user_prompt, get_llm  # noqa: E402

st.set_page_config(page_title="RAG · Demonstrações Flamengo", page_icon="📊", layout="centered")


@st.cache_resource(show_spinner="Carregando modelo de embedding...")
def _warm():
    from src.embed_index import get_embed_model
    get_embed_model()


def _available_models() -> list[str]:
    import httpx
    try:
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json()
        return sorted(m["name"] for m in tags.get("models", []))
    except Exception:
        return [config.OLLAMA_MODEL]


st.title("📊 Demonstrações financeiras do Flamengo")
st.caption("RAG 100% local — busca nos PDFs de 2022–2026 e responde só com base neles.")

_warm()

with st.sidebar:
    st.header("Configuração")
    mode = st.selectbox("Retrieval", MODES, index=MODES.index("bm25"),
                        help="bm25 = busca lexical (melhor aqui). hybrid = lexical + semântica.")
    k = st.slider("Trechos recuperados (k)", 1, 6, config.GENERATION_TOP_K)
    models = _available_models()
    model = st.selectbox("Modelo (Ollama)", models,
                         index=models.index(config.OLLAMA_MODEL) if config.OLLAMA_MODEL in models else 0)
    strategy = st.radio("Chunking", ["fixed", "structural"], horizontal=True)
    st.divider()
    st.caption("O ano é inferido da pergunta (ex: \"...em 2024?\"). "
               "Force abaixo se precisar.")
    year_override = st.text_input("Forçar exercício", placeholder="ex: 2024")

question = st.text_input("Sua pergunta", placeholder="Qual foi a receita operacional líquida em 2023?")

if st.button("Perguntar", type="primary", disabled=not question):
    year = year_override.strip() or guess_year(question)

    with st.spinner(f"Buscando ({mode}" + (f", ano {year}" if year else "") + ")..."):
        hits = retrieve(question, strategy, mode, k, year)

    if not hits:
        st.warning("Nenhum trecho recuperado.")
        st.stop()

    messages_preview = build_user_prompt(question, build_context(hits))

    with st.spinner(f"Gerando resposta com {model} (pode levar ~1 min em CPU)..."):
        from llama_index.core.llms import ChatMessage, MessageRole
        t0 = time.time()
        resp = get_llm(model).chat([
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=messages_preview),
        ])
        dt = time.time() - t0

    st.markdown("### Resposta")
    st.write(str(resp.message.content).strip())
    st.caption(f"{dt:.0f}s · {mode} · k={k} · {model}" + (f" · ano {year}" if year else ""))

    st.markdown("### Trechos usados (confira a fundamentação)")
    for i, h in enumerate(hits):
        m = h.node.metadata
        score = f"{h.score:.3f}" if h.score is not None else "—"
        with st.expander(f"#{i + 1} · {m.get('source', '?')} · {m.get('header_path', '-')} · score {score}"):
            st.text(h.node.get_content())
