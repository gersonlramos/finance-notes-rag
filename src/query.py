"""
Milestone 3 — Retrieval (ainda SEM LLM)
=======================================

Cobre a **Etapa 5** do pipeline. Dada uma pergunta, buscamos os chunks mais
relevantes no índice — e só. Nenhuma resposta é gerada aqui. O objetivo é você
VER o que o retrieval traz e entender por que às vezes ele erra.

Três formas de buscar (o `--compare` roda as três lado a lado):

  vector  — busca DENSA: embedda a pergunta, acha os vetores mais próximos.
            Entende paráfrase e sinônimo. Borra termos técnicos exatos.

  bm25    — busca ESPARSA / LEXICAL: pontua por frequência de termos (com
            stemming em português). Acerta "contingências", "PROFUT", siglas.
            Não entende sinônimo nenhum.

  hybrid  — roda as duas e funde os rankings (Reciprocal Rank Fusion). Um chunk
            bem posicionado nas duas listas sobe.

Filtro por metadado: `--year 2024` restringe a busca aos chunks cujo documento
é do exercício de 2024 (campo `doc_year`, definido na ingestão).

Uso:
    python -m src.query "qual a provisão para contingências em 2024?" --compare
    python -m src.query "receita com direitos de transmissão" --mode hybrid --k 8
    python -m src.query "endividamento líquido" --mode vector --year 2023
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import Stemmer
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever
from rich.console import Console

from src import config
from src.embed_index import open_index

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Milestone 3 não usa LLM. O QueryFusionRetriever tenta resolver um LLM no
# __init__ mesmo com num_queries=1; MockLLM evita a busca pelo pacote da OpenAI.
Settings.llm = MockLLM()

console = Console()

MODES = ("vector", "bm25", "hybrid")

_YEAR_RE = re.compile(r"\b(20[12]\d)\b")


def guess_year(question: str) -> str | None:
    """Extrai um ano 20xx da pergunta ('...em 2024?' -> '2024'). Usado como
    filtro automático — a avaliação mostrou que filtrar por exercício é o que
    mais melhora o retrieval nesses documentos multi-ano."""
    anos = _YEAR_RE.findall(question)
    return anos[-1] if anos else None


def _year_filter(year: str | None) -> MetadataFilters | None:
    if not year:
        return None
    return MetadataFilters(filters=[
        MetadataFilter(key="doc_year", value=year, operator=FilterOperator.EQ)
    ])


def load_nodes(strategy: str) -> list[TextNode]:
    """Chunks do jsonl, em memória — o BM25 precisa do texto de todos eles."""
    path = config.CHUNKS_DIR / f"{strategy}.jsonl"
    nodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        nodes.append(TextNode(id_=o["id"], text=o["text"], metadata=o["metadata"]))
    return nodes


def get_retriever(mode: str, strategy: str, k: int, year: str | None):
    filters = _year_filter(year)

    if mode == "vector":
        return open_index(strategy).as_retriever(similarity_top_k=k, filters=filters)

    if mode == "bm25":
        return BM25Retriever.from_defaults(
            nodes=load_nodes(strategy),
            similarity_top_k=k,
            language="portuguese",
            stemmer=Stemmer.Stemmer("portuguese"),
            filters=filters,
        )

    if mode == "hybrid":
        vec = open_index(strategy).as_retriever(similarity_top_k=k, filters=filters)
        bm25 = BM25Retriever.from_defaults(
            nodes=load_nodes(strategy),
            similarity_top_k=k,
            language="portuguese",
            stemmer=Stemmer.Stemmer("portuguese"),
            filters=filters,
        )
        return QueryFusionRetriever(
            retrievers=[vec, bm25],
            mode="reciprocal_rerank",   # RRF: soma 1/(rank+c) de cada lista
            similarity_top_k=k,
            num_queries=1,              # 1 = não usa LLM para gerar variações
            use_async=False,
        )

    raise ValueError(mode)


def retrieve(question: str, strategy: str, mode: str, k: int, year: str | None) -> list[NodeWithScore]:
    return get_retriever(mode, strategy, k, year).retrieve(question)


def print_hits(hits: list[NodeWithScore], title: str) -> None:
    console.print(f"\n[bold]{title}[/bold]  ({len(hits)} chunks)")
    for i, h in enumerate(hits):
        m = h.node.metadata
        score = f"{h.score:.3f}" if h.score is not None else "  -  "
        console.print(
            f"[cyan]#{i + 1}  score={score}  "
            f"{m.get('source', '?')[:38]}  ano={m.get('doc_year', '?')}  "
            f"seção: {str(m.get('header_path', '-'))[:50]}[/cyan]"
        )
        txt = " ".join(h.node.get_content(metadata_mode=MetadataMode.NONE).split())
        console.print(f"   {txt[:240]}{'…' if len(txt) > 240 else ''}")


def compare(question: str, strategy: str, k: int, year: str | None) -> None:
    console.print(f"[bold yellow]pergunta:[/bold yellow] {question}")
    if year:
        console.print(f"[dim]filtro: doc_year == {year}[/dim]")
    for mode in MODES:
        print_hits(retrieve(question, strategy, mode, k, year), f"MODO: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval sem LLM (Milestone 3)")
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=["fixed", "structural"], default="fixed")
    ap.add_argument("--mode", choices=MODES, default="bm25")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--year", help="filtra por exercício do documento (ex: 2024)")
    ap.add_argument("--compare", action="store_true", help="roda os 3 modos lado a lado")
    args = ap.parse_args()

    if args.compare:
        compare(args.question, args.strategy, args.k, args.year)
    else:
        console.print(f"[bold yellow]pergunta:[/bold yellow] {args.question}")
        hits = retrieve(args.question, args.strategy, args.mode, args.k, args.year)
        print_hits(hits, f"MODO: {args.mode}")


if __name__ == "__main__":
    main()
