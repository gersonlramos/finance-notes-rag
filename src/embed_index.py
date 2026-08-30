"""
Milestone 2 — Embeddings + Indexação
====================================

Cobre as **Etapas 3 e 4** do pipeline (ver notes/guia-conceitual-rag.md):

    chunks (jsonl)  --[BGE-M3]-->  vetores  -->  ChromaDB (chroma_db/)

O que acontece aqui:
  1. Carrega os chunks de data/chunks/<estrategia>.jsonl
  2. O modelo BGE-M3 converte cada chunk num vetor de 1024 números que
     representa o SIGNIFICADO do texto. Chunks com sentido parecido ficam
     próximos nesse espaço de 1024 dimensões.
  3. Cada (id, vetor, texto, metadados) vai para uma coleção do ChromaDB,
     que monta um índice (HNSW) para busca por similaridade em milissegundos.
  4. O índice fica salvo em disco. Isto roda UMA vez (ou quando os chunks
     mudam); o retrieval do Milestone 3 só carrega.

Uma coleção por estratégia (chunks_structural, chunks_fixed) para podermos
comparar as duas no Milestone 6.

Uso:
    python -m src.embed_index --strategy structural --rebuild
    python -m src.embed_index --strategy structural --probe "qual foi o saldo de caixa em 2023?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

_embed_model: FastEmbedEmbedding | None = None


def get_embed_model() -> FastEmbedEmbedding:
    """
    Carrega o modelo de embedding uma vez por processo, via fastembed (ONNX,
    roda em CPU). Na 1ª execução baixa o modelo (~230 MB) para o cache local.
    """
    global _embed_model
    if _embed_model is None:
        console.print(f"[dim]carregando {config.EMBED_MODEL} (ONNX/CPU)...[/dim]")
        _embed_model = FastEmbedEmbedding(
            model_name=config.EMBED_MODEL,
            embed_batch_size=config.EMBED_BATCH_SIZE,
        )
    return _embed_model


def load_chunks(strategy: str, limit: int | None = None) -> list[TextNode]:
    """Lê data/chunks/<estrategia>.jsonl e reconstrói os nós do LlamaIndex."""
    path = config.CHUNKS_DIR / f"{strategy}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} não existe. Rode:  python -m src.ingest --strategy {strategy}")

    nodes: list[TextNode] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        nodes.append(TextNode(
            id_=obj["id"],
            text=obj["text"],
            metadata=obj["metadata"],
            # não deixa o texto dos metadados "vazar" para dentro do embedding
            excluded_embed_metadata_keys=list(obj["metadata"].keys()),
            excluded_llm_metadata_keys=[],
        ))
        if limit and len(nodes) >= limit:
            break
    return nodes


def build_index(strategy: str, rebuild: bool = False, limit: int | None = None) -> VectorStoreIndex:
    nodes = load_chunks(strategy, limit)
    console.print(f"[green]{len(nodes)}[/green] chunks de '{strategy}'")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    name = config.collection_name(strategy)
    if rebuild:
        try:
            client.delete_collection(name)
            console.print(f"[yellow]coleção '{name}' apagada (rebuild)[/yellow]")
        except Exception:
            pass
    # hnsw:space=cosine -> distância de cosseno (ignora a norma dos vetores;
    # o MiniLM do fastembed não devolve vetores normalizados). O score do
    # retrieval vira 1 - distância.
    collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    t0 = time.time()
    index = VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=get_embed_model(),
        show_progress=True,
    )
    dt = time.time() - t0

    console.print(
        f"[bold green]indexado[/bold green] em {dt:.0f}s "
        f"({dt / max(len(nodes), 1):.2f}s/chunk) | "
        f"coleção '{name}' tem {collection.count()} vetores de dim {config.EMBED_DIM}"
    )
    return index


def open_index(strategy: str) -> VectorStoreIndex:
    """Reabre um índice já persistido, sem re-embeddar (usado no Milestone 3)."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(config.collection_name(strategy))
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())


def probe(strategy: str, question: str, k: int = 3) -> None:
    """Busca rápida para PROVAR que o índice funciona e ver o retrieval cru."""
    index = open_index(strategy)
    retriever = index.as_retriever(similarity_top_k=k)
    console.print(f"\n[bold]pergunta:[/bold] {question}")
    for i, hit in enumerate(retriever.retrieve(question)):
        meta = hit.node.metadata
        console.print(
            f"\n[cyan]#{i + 1}  score={hit.score:.3f}  "
            f"fonte: {meta.get('source', '?')}  "
            f"seção: {meta.get('header_path', '-')}[/cyan]"
        )
        txt = hit.node.get_content(metadata_mode=MetadataMode.NONE)
        console.print(txt[:500] + (" [...]" if len(txt) > 500 else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Embeddings + indexação (Milestone 2)")
    ap.add_argument("--strategy", choices=["fixed", "structural"], default="fixed")
    ap.add_argument("--rebuild", action="store_true", help="apaga e reconstrói a coleção")
    ap.add_argument("--limit", type=int, help="indexar só os N primeiros chunks (teste)")
    ap.add_argument("--probe", metavar="PERGUNTA", help="após indexar, faz uma busca de teste")
    args = ap.parse_args()

    if args.probe and not args.rebuild:
        probe(args.strategy, args.probe)
    else:
        build_index(args.strategy, rebuild=args.rebuild, limit=args.limit)
        if args.probe:
            probe(args.strategy, args.probe)


if __name__ == "__main__":
    main()
