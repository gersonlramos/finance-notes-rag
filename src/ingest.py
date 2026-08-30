"""
Milestone 1 — Ingestão + Chunking
=================================

Este script cobre as **Etapas 1 e 2** do pipeline (ver
`notes/guia-conceitual-rag.md`):

    documentos  ->  carregar (texto + metadados)  ->  dividir em chunks

Ainda NÃO há embeddings nem banco vetorial. O objetivo aqui é você *ver*
como os documentos são cortados e comparar duas estratégias de chunking.

Uso:
    python -m src.ingest --source data/sample_docs --strategy fixed
    python -m src.ingest --source data/sample_docs --strategy structural --show 5

Saída:
    - um resumo no terminal (nº de docs, nº de chunks, distribuição de tamanho)
    - alguns chunks de amostra
    - um arquivo data/chunks/<strategy>.jsonl com todos os chunks, para inspeção
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import median

# Console do Windows costuma vir em cp1252; forçamos UTF-8 para não quebrar
# na hora de imprimir acentos e caracteres de caixa.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode
from rich.console import Console
from rich.table import Table

from src import config

console = Console()

# Extensões que sabemos carregar. PDF é tratado à parte (precisa de parser).
TEXT_EXTS = {".md", ".markdown", ".txt"}
PDF_EXTS = {".pdf"}


# ======================================================================
#  ETAPA 1 — CARREGAR
# ======================================================================
def load_documents(source_dir: Path, do_clean: bool = True,
                   linearize_tables: bool = False) -> list[Document]:
    """
    Lê todos os arquivos suportados de `source_dir` e devolve uma lista de
    `Document` do LlamaIndex.

    Um `Document` = o conteúdo bruto de UMA fonte (um arquivo inteiro) +
    metadados. Ele ainda não foi cortado; isso é a Etapa 2.

    Decisões importantes aqui:
      - PDF é convertido para Markdown (via pymupdf4llm) para PRESERVAR os
        títulos. Sem isso, a estratégia "structural" não teria como saber
        onde uma seção começa e termina.
      - Guardamos metadados (`source`, `filetype`). Eles viajam junto com
        cada chunk e serão usados para citar a fonte na resposta final.
    """
    files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file()
        # ignora pastas tipo data/raw_docs/_ocr_pendente/ (PDFs escaneados)
        and not any(part.startswith(config.IGNORE_DIR_PREFIX) for part in p.relative_to(source_dir).parts[:-1])
    )
    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado em {source_dir}")

    documents: list[Document] = []
    for path in files:
        ext = path.suffix.lower()

        if ext in TEXT_EXTS:
            text = path.read_text(encoding="utf-8", errors="replace")

        elif ext in PDF_EXTS:
            text = _pdf_to_markdown(path)

        else:
            console.print(f"[dim]ignorando {path.name} (extensão {ext} não suportada)[/dim]")
            continue

        if not text.strip():
            console.print(f"[yellow]aviso:[/yellow] {path.name} resultou em texto vazio")
            continue

        if do_clean:
            # Etapa 1.5: limpa o Markdown extraído (src/clean.py) e lineariza
            # as tabelas em frases por linha (src/tables.py, Milestone 6).
            from src import clean
            text = clean.clean_markdown(text)
            if linearize_tables:
                from src import tables
                text = tables.linearize(text)
            clean.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
            (clean.CLEAN_DIR / f"{path.stem}.md").write_text(text, encoding="utf-8")

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": str(path.relative_to(source_dir)),
                    "filetype": ext.lstrip("."),
                    # ano principal do documento (extraído do nome do arquivo).
                    # Serve para filtrar a busca por exercício no Milestone 3.
                    "doc_year": _guess_year(path.stem),
                },
            )
        )

    console.print(f"[green]Carregados[/green] {len(documents)} documento(s) de {source_dir}")
    return documents


def _guess_year(name: str) -> str:
    """Último ano 20xx no nome do arquivo ('...2024.pdf' -> '2024'). '' se não achar."""
    anos = re.findall(r"20\d{2}", name)
    return anos[-1] if anos else ""


def _pdf_to_markdown(path: Path) -> str:
    """
    Converte um PDF em Markdown, com cache em data/extracted/.

    A extração é a etapa LENTA (dezenas de segundos por PDF grande) e não muda
    entre experimentos de chunking. Por isso ela é separada e cacheada:
    extrai uma vez, e todo experimento de chunking reusa o Markdown.

    pymupdf4llm reconstrói tabelas como tabelas Markdown e lida com layout
    multi-coluna — essencial para demonstrativos financeiros (balanço, DRE).
    Ainda assim, abra o .md gerado e confira: gráficos viram "picture text"
    ruidoso, e cabeçalhos/rodapés se repetem.
    """
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.EXTRACTED_DIR / f"{path.stem}.md"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return cache.read_text(encoding="utf-8")

    from src import ocr

    if ocr.needs_ocr(path):
        # PDF sem camada de texto (2024, 2025) -> OCR página a página.
        console.print(f"[yellow]OCR[/yellow] {path.name} (sem texto extraível — vai demorar)...")
        md = ocr.ocr_pdf_to_markdown(path)
    else:
        try:
            import pymupdf4llm
        except ImportError:
            raise SystemExit(
                "PDF encontrado mas 'pymupdf4llm' não está instalado.\n"
                "Instale as deps do Milestone 1:  pip install -r requirements.txt"
            )
        console.print(f"[dim]extraindo {path.name} (pode demorar)...[/dim]")
        # use_ocr=False: o OCR é feito só pelo nosso src/ocr.py (controle explícito).
        md = pymupdf4llm.to_markdown(str(path), use_ocr=False, show_progress=False)

    cache.write_text(md, encoding="utf-8")
    return md


# ======================================================================
#  ETAPA 2 — CHUNKING
# ======================================================================
def build_parser(strategy: str):
    """
    Devolve o "node parser" do LlamaIndex correspondente à estratégia.

    LlamaIndex chama cada chunk de **Node**. Um parser recebe `Document`s e
    devolve `Node`s.

    - "fixed"      -> SentenceSplitter: junta frases até chegar perto de
                      CHUNK_SIZE tokens, com CHUNK_OVERLAP tokens repetidos
                      no começo do próximo chunk. Previsível e uniforme.

    - "structural" -> MarkdownNodeParser: corta nos títulos (#, ##, ###).
                      Cada chunk é uma seção inteira -> contexto coeso, mas
                      tamanhos MUITO desiguais. Em seguida passamos um
                      SentenceSplitter só para reparticionar seções gigantes.
    """
    if strategy == "fixed":
        return SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    if strategy == "structural":
        return MarkdownNodeParser()
    raise ValueError(f"estratégia desconhecida: {strategy}")


def chunk_documents(documents: list[Document], strategy: str) -> list[BaseNode]:
    parser = build_parser(strategy)
    nodes = parser.get_nodes_from_documents(documents, show_progress=False)

    if strategy == "structural":
        # MarkdownNodeParser não subdivide seções grandes. Reparticionamos
        # só as que passam do limite, preservando os metadados de seção.
        splitter = SentenceSplitter(
            chunk_size=config.STRUCTURAL_MAX_TOKENS,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(
            [Document(text=n.get_content(), metadata=n.metadata) for n in nodes],
            show_progress=False,
        )

    return _merge_tiny_nodes(nodes)


def _merge_tiny_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Funde chunks minúsculos (< MIN_CHUNK_TOKENS) no vizinho do mesmo documento.
    Um título solto ("### 4. Contas a receber") sozinho num chunk não ajuda o
    retrieval — colado ao parágrafo seguinte, ajuda.
    """
    enc = tiktoken.get_encoding(config.TOKEN_ENCODING)
    merged: list[BaseNode] = []
    carry = ""  # chunk pequeno esperando para ser anexado ao próximo
    for node in nodes:
        text = f"{carry}\n{node.get_content()}".strip() if carry else node.get_content()
        carry = ""
        same_src = merged and merged[-1].metadata.get("source") == node.metadata.get("source")
        if len(enc.encode(text)) < config.MIN_CHUNK_TOKENS:
            if same_src:
                merged[-1].text = f"{merged[-1].get_content()}\n{text}".strip()
            else:
                carry = text
            continue
        node.text = text
        merged.append(node)
    if carry and merged:
        merged[-1].text = f"{merged[-1].get_content()}\n{carry}".strip()
    return merged


# ======================================================================
#  INSPEÇÃO — como você aprende olhando o resultado
# ======================================================================
def token_len(text: str, encoder) -> int:
    return len(encoder.encode(text))


def report(nodes: list[BaseNode], strategy: str, show: int) -> None:
    encoder = tiktoken.get_encoding(config.TOKEN_ENCODING)
    sizes = [token_len(n.get_content(), encoder) for n in nodes]

    table = Table(title=f"Estrategia: {strategy}  -  {len(nodes)} chunks")
    table.add_column("métrica")
    table.add_column("tokens", justify="right")
    table.add_row("mínimo", str(min(sizes)))
    table.add_row("mediana", str(int(median(sizes))))
    table.add_row("média", str(int(sum(sizes) / len(sizes))))
    table.add_row("máximo", str(max(sizes)))
    table.add_row("total", str(sum(sizes)))
    console.print(table)

    # Chunks muito curtos (<20 tokens) costumam ser lixo: títulos soltos,
    # rodapés, linhas de tabela quebradas. Sinal de problema na Etapa 1.
    tiny = sum(1 for s in sizes if s < 20)
    if tiny:
        console.print(f"[yellow]{tiny} chunk(s) com < 20 tokens[/yellow] — provável ruído de extração")

    # linha fácil de comparar entre execuções
    console.print(f"RESUMO {strategy}: {len(nodes)} chunks | "
                  f"mediana {int(median(sizes))} tok | <20tok: {tiny}")

    console.print(f"\n[bold]Amostra ({min(show, len(nodes))} chunks):[/bold]")
    for i, node in enumerate(nodes[:show]):
        content = node.get_content()
        console.print(f"\n[cyan]--- chunk {i} | {token_len(content, encoder)} tokens |"
                      f" fonte: {node.metadata.get('source', '?')} ---[/cyan]")
        header = node.metadata.get("header_path") or node.metadata.get("Header 1")
        if header:
            console.print(f"[dim]secao: {header}[/dim]")
        preview = content if len(content) < 600 else content[:600] + " [...]"
        console.print(preview)


def dump_jsonl(nodes: list[BaseNode], strategy: str) -> Path:
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CHUNKS_DIR / f"{strategy}.jsonl"
    encoder = tiktoken.get_encoding(config.TOKEN_ENCODING)
    with out.open("w", encoding="utf-8") as fh:
        for node in nodes:
            content = node.get_content()
            fh.write(json.dumps({
                "id": node.node_id,
                "text": content,
                "n_tokens": token_len(content, encoder),
                "metadata": node.metadata,
            }, ensure_ascii=False) + "\n")
    console.print(f"\n[green]Chunks salvos em[/green] {out}")
    return out


# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestão + chunking (Milestone 1)")
    ap.add_argument("--source", type=Path, default=config.SAMPLE_DOCS_DIR,
                    help="pasta com os documentos (default: data/sample_docs)")
    ap.add_argument("--strategy", choices=["fixed", "structural"], default="fixed")
    ap.add_argument("--show", type=int, default=3, help="quantos chunks de amostra imprimir")
    ap.add_argument("--no-dump", action="store_true", help="não escrever o .jsonl")
    ap.add_argument("--no-clean", action="store_true", help="pular a limpeza (Etapa 1.5)")
    ap.add_argument("--linearize", action="store_true",
                    help="linearizar tabelas em frases (experimento M6 — piorou a acurácia)")
    args = ap.parse_args()

    documents = load_documents(args.source, do_clean=not args.no_clean,
                               linearize_tables=args.linearize)
    nodes = chunk_documents(documents, args.strategy)
    report(nodes, args.strategy, args.show)
    if not args.no_dump:
        dump_jsonl(nodes, args.strategy)


if __name__ == "__main__":
    main()
