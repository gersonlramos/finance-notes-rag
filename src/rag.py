"""
Milestone 4 — Geração da resposta (Etapa 6)
===========================================

    retrieval (Milestone 3)  ->  monta prompt com o contexto  ->  LLM local gera

O pipeline RAG completo, enfim: você pergunta em linguagem natural, o sistema
busca nos documentos e o LLM redige a resposta **fundamentada só no contexto
recuperado** — não no conhecimento geral do modelo.

O LLM roda no Ollama (local, CPU, custo zero). O que impede a alucinação:
  - o PROMPT DE SISTEMA manda usar só o contexto e permitir "não sei";
  - temperatura baixa (config.GENERATION_TEMPERATURE);
  - as fontes de cada trecho vão no prompt, para o modelo poder citar.

Uso:
    python -m src.rag "qual foi o superávit do Flamengo em 2023?"
    python -m src.rag "provisão para contingências em 2024" --mode bm25 --year 2024
    python -m src.rag "receita de bilheteria em 2023" --show-context
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import httpx
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from rich.console import Console

from src import config
from src.query import MODES, guess_year, retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

SYSTEM_PROMPT = """Você é um analista que responde perguntas sobre as demonstrações \
financeiras do Clube de Regatas do Flamengo usando SOMENTE o contexto fornecido.

Regras:
- Baseie-se apenas nos trechos do contexto. Não use conhecimento externo.
- Se a resposta não estiver EXPLÍCITA no contexto, responda exatamente: \
"Não encontrei essa informação nos documentos." Não deduza, não estime.
- NÃO faça cálculos (somas, subtrações, variações). Apenas transcreva os \
valores que aparecem.
- Os valores estão em milhares de reais, salvo indicação em contrário \
(ex: "234.487" = R$ 234,487 milhões). Reproduza o número como aparece e diga a unidade.
- Ao final de cada afirmação, cite a fonte entre colchetes: [arquivo, seção].
- Seja conciso: 1 a 3 frases.
"""


def build_context(hits) -> str:
    blocos = []
    for i, h in enumerate(hits):
        m = h.node.metadata
        cab = f"[trecho {i + 1} | {m.get('source', '?')} | ano {m.get('doc_year', '?')} | {m.get('header_path', '-')}]"
        txt = h.node.get_content()
        if len(txt) > config.CONTEXT_CHARS_PER_CHUNK:
            txt = txt[:config.CONTEXT_CHARS_PER_CHUNK] + " […]"
        blocos.append(f"{cab}\n{txt}")
    return "\n\n".join(blocos)


def build_user_prompt(question: str, context: str) -> str:
    return f"Contexto:\n{context}\n\n---\nPergunta: {question}"


def is_claude(model: str) -> bool:
    return model.lower().startswith("claude")


# Modelos Claude que funcionam com o stack atual. Os modelos 5 não aceitam
# mais `temperature` na API, e o wrapper do LlamaIndex só remove esse
# parâmetro para esta lista — passar temperature em qualquer outro quebra
# ("Messages.create() got an unexpected keyword argument 'temperature'").
CLAUDE_MODELS = ("claude-sonnet-5", "claude-opus-5", "claude-opus-4-8")


def get_llm(model: str) -> LLM:
    """
    Devolve o LLM (interface LlamaIndex — .chat()/.stream_chat()).
    - "claude-*"  -> API da Anthropic (precisa de ANTHROPIC_API_KEY). NÃO é local
                     nem grátis; é a válvula de escape para qualidade máxima.
    - resto       -> Ollama local (padrão, custo zero).
    """
    if is_claude(model):
        from llama_index.llms.anthropic import Anthropic

        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "ANTHROPIC_API_KEY não definida. Coloque a chave no arquivo .env "
                "(ANTHROPIC_API_KEY=sk-ant-...) para usar um modelo Claude."
            )
        if model not in CLAUDE_MODELS:
            raise SystemExit(
                f"Modelo Claude '{model}' não suportado com o stack atual. "
                f"Use um destes: {', '.join(CLAUDE_MODELS)}."
            )
        # sem `temperature`: os modelos Claude 5 não aceitam mais esse parâmetro.
        return Anthropic(model=model, api_key=key, max_tokens=config.ANTHROPIC_MAX_TOKENS)

    opts = {"num_predict": config.OLLAMA_NUM_PREDICT}
    # Modelos >= 7B: forçar CPU puro. A GeForce MX110 (2GB) só cabe uma fração
    # das camadas e o Ollama fica trocando dados CPU<->GPU — fica MAIS lento que
    # CPU sozinha (7B: ~15min com offload vs ~3min sem).
    if re.search(r"\b(7|8|9|1[0-9])b\b", model.lower()):
        opts["num_gpu"] = 0
    return Ollama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.OLLAMA_TIMEOUT,
        temperature=config.GENERATION_TEMPERATURE,
        # sem isto o Ollama usa num_ctx pequeno e CORTA o contexto em silêncio
        context_window=config.OLLAMA_CONTEXT_WINDOW,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        additional_kwargs=opts,
    )


def _messages(question: str, hits) -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_user_prompt(question, build_context(hits))),
    ]


def run_stream(question: str, *, strategy: str = "fixed", mode: str = "bm25",
               k: int | None = None, year: str | None = None, model: str | None = None,
               auto_year: bool = True):
    """
    Igual a run(), mas devolve (hits, gerador_de_tokens). O app Streamlit usa
    para exibir a resposta se formando — a percepção de velocidade melhora muito.
    """
    k = k or config.GENERATION_TOP_K
    model = model or config.OLLAMA_MODEL
    if year is None and auto_year:
        year = guess_year(question)
    _check_backend(model)
    hits = retrieve(question, strategy, mode, k, year)
    if not hits:
        return [], iter(())

    def tokens():
        for chunk in get_llm(model).stream_chat(_messages(question, hits)):
            yield chunk.delta or ""

    return hits, tokens()


def _check_backend(model: str) -> None:
    if is_claude(model):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY não definida (veja .env.example).")
        return
    try:
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except Exception:
        raise SystemExit(
            "Ollama não está respondendo em " + config.OLLAMA_BASE_URL + ".\n"
            "Instale (winget install Ollama.Ollama) e deixe o serviço rodando."
        )
    have = {m["name"] for m in tags.get("models", [])}
    if model not in have and f"{model}:latest" not in have:
        raise SystemExit(f"Modelo '{model}' não baixado. Rode:  ollama pull {model}")


def run(question: str, *, strategy: str = "fixed", mode: str = "bm25",
        k: int | None = None, year: str | None = None,
        model: str | None = None, auto_year: bool = True) -> dict:
    """
    Executa o pipeline e DEVOLVE o resultado (não imprime). Usado pelo
    evaluate.py (Milestone 5) e pelo app Streamlit (Milestone 7).

    auto_year: se `year` não vier, tenta extrair o exercício da própria
    pergunta ("...em 2024?" -> filtra doc_year=2024). A avaliação mostrou que
    isto é o que mais melhora o retrieval nesta base multi-ano.
    """
    k = k or config.GENERATION_TOP_K
    model = model or config.OLLAMA_MODEL
    if year is None and auto_year:
        year = guess_year(question)
    _check_backend(model)

    hits = retrieve(question, strategy, mode, k, year)
    contexts = [h.node.get_content() for h in hits]
    if not hits:
        return {"question": question, "answer": "", "contexts": [], "hits": [], "seconds": 0.0}

    t0 = time.time()
    resp = get_llm(model).chat(_messages(question, hits))
    return {
        "question": question,
        "answer": str(resp.message.content).strip(),
        "contexts": contexts,
        "hits": hits,
        "seconds": time.time() - t0,
    }


def answer(question: str, *, strategy: str, mode: str, k: int, year: str | None,
           model: str, show_context: bool = False) -> None:
    eff_year = year or guess_year(question)
    console.print(f"[bold yellow]pergunta:[/bold yellow] {question}")
    console.print(f"[dim]{mode} · k={k}" + (f" · ano={eff_year}" if eff_year else "") + f" · {model}[/dim]\n")

    if show_context:
        hits = retrieve(question, strategy, mode, k, eff_year)
        console.print(f"[dim]{build_context(hits)}[/dim]\n{'─' * 60}")

    r = run(question, strategy=strategy, mode=mode, k=k, year=year, model=model)
    if not r["hits"]:
        console.print("[yellow]Nenhum trecho recuperado — nada a responder.[/yellow]")
        return

    console.print(r["answer"])
    console.print(f"\n[dim]({r['seconds']:.0f}s)  trechos usados:[/dim]")
    for i, h in enumerate(r["hits"]):
        m = h.node.metadata
        console.print(f"  [dim][trecho {i + 1}] {m.get('source', '?')} — {m.get('header_path', '-')}[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline RAG completo (Milestone 4)")
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=["fixed", "structural"], default="fixed")
    ap.add_argument("--mode", choices=MODES, default="bm25")
    ap.add_argument("--k", type=int, default=config.GENERATION_TOP_K)
    ap.add_argument("--year", help="força o filtro de exercício (senão é inferido da pergunta)")
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--show-context", action="store_true", help="imprime o contexto enviado ao LLM")
    args = ap.parse_args()

    answer(args.question, strategy=args.strategy, mode=args.mode, k=args.k,
           year=args.year, model=args.model, show_context=args.show_context)


if __name__ == "__main__":
    main()
