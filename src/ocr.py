"""
OCR para PDFs sem camada de texto (Etapa 1, casos difíceis)
==========================================================

Os PDFs de 2024 e 2025 não têm texto extraível:
  - 2024: cada página é uma imagem raster (render limpo da página)
  - 2025: o texto foi convertido em contornos vetoriais

Estratégia: renderizar cada página em alta resolução e passar pelo Tesseract
(motor de OCR local, offline, gratuito) com o pacote de idioma português.

O Tesseract devolve cada PALAVRA com sua caixa (coordenadas). A parte
não-trivial é reconstruir a ORDEM DE LEITURA e o alinhamento de colunas das
tabelas a partir dessas caixas — é o que `_reconstruct_page` faz.

Pré-requisito: Tesseract instalado no sistema, com o idioma `por`.
    winget install UB-Mannheim.TesseractOCR      (ou: choco install tesseract)
O binário é localizado automaticamente em `_configure_tesseract()`.

Limitação conhecida: tabelas muito largas podem ter colunas desalinhadas
(o Tesseract não tem reconhecimento de tabela nativo; a reconstrução por
coordenadas é aproximada).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
import pytesseract

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# O PDF de 2025 tem estrutura interna irregular; o parser MuPDF cospe centenas
# de "syntax error" no stderr que não afetam o resultado. Silencia.
pymupdf.TOOLS.mupdf_display_errors(False)

# Abaixo deste nº médio de caracteres por página, consideramos que o PDF
# não tem camada de texto e precisa de OCR.
MIN_CHARS_PER_PAGE = 200

# Resolução de render antes do OCR. 300 DPI é o mínimo recomendado para OCR
# de texto tipográfico; abaixo disso a taxa de erro sobe rápido.
OCR_DPI = 300

# psm 4 = "assume uma única coluna de texto de tamanhos variados" — bom para
# páginas de nota explicativa (texto + tabelas numa coluna só).
# O por.traineddata fica em models/tessdata/ (o instalador do Tesseract via
# winget só traz inglês). Apontamos via env var TESSDATA_PREFIX em _ready().
TESSERACT_CONFIG = "--oem 1 --psm 4 -c preserve_interword_spaces=1"
TESSERACT_LANG = "por"

# Locais comuns de instalação do Tesseract no Windows.
_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    str(Path.home() / r"AppData\Local\Tesseract-OCR\tesseract.exe"),
]

# Linhas de cabeçalho/rodapé que se repetem em toda página — puro ruído.
JUNK_PATTERNS = [
    re.compile(r"docusign envelope id", re.I),
    re.compile(r"^cr\.?flamengo", re.I),
    re.compile(r"^\d{1,3}$"),  # número de página solto
    re.compile(r"^relat[oó]rio\s+de\s+gest[aã]o", re.I),
]

# Uma linha tipo "3. Caixa e equivalentes de caixa" é um título de nota
# explicativa. Promovemos a header Markdown para o chunking estrutural funcionar
# também nos docs de OCR.
NOTE_HEADING = re.compile(r"^(\d{1,2})\.\s+([A-ZÀ-Ú][^\n]{3,90})$")


def _configure_tesseract() -> None:
    """Localiza o binário do Tesseract; erro claro se não estiver instalado."""
    found = shutil.which("tesseract")
    if not found:
        for p in _WINDOWS_PATHS:
            if Path(p).exists():
                found = p
                break
    if not found:
        raise SystemExit(
            "Tesseract não encontrado. Instale com:\n"
            "    winget install UB-Mannheim.TesseractOCR\n"
            "e garanta o idioma 'por' (o instalador UB-Mannheim já inclui)."
        )
    pytesseract.pytesseract.tesseract_cmd = found


def needs_ocr(pdf_path: Path, min_chars: int = MIN_CHARS_PER_PAGE) -> bool:
    """True se o PDF tem pouca ou nenhuma camada de texto."""
    doc = pymupdf.open(pdf_path)
    total = sum(len(page.get_text("text").strip()) for page in doc)
    avg = total / max(doc.page_count, 1)
    doc.close()
    return avg < min_chars


@lru_cache(maxsize=1)
def _ready() -> bool:
    _configure_tesseract()
    traineddata = config.TESSDATA_DIR / f"{TESSERACT_LANG}.traineddata"
    if not traineddata.exists():
        raise SystemExit(
            f"Falta o modelo de idioma: {traineddata}\n"
            f"Baixe com:\n"
            f'    curl -L -o "{traineddata}" '
            f"https://github.com/tesseract-ocr/tessdata_best/raw/main/{TESSERACT_LANG}.traineddata"
        )
    os.environ["TESSDATA_PREFIX"] = str(config.TESSDATA_DIR)
    return True


def ocr_pdf_to_markdown(pdf_path: Path, dpi: int = OCR_DPI) -> str:
    """
    Converte um PDF escaneado em Markdown, página por página.

    Híbrido: se uma página TEM texto (ex: cabeçalhos reais em 2024), usa esse
    texto direto; só passa por OCR o que não tem.
    """
    _ready()
    doc = pymupdf.open(pdf_path)
    parts: list[str] = [f"# {pdf_path.stem}\n"]

    for i, page in enumerate(doc):
        native = page.get_text("text").strip()
        if len(native) >= MIN_CHARS_PER_PAGE:
            body = native
        else:
            body = _ocr_page(page, dpi)

        body = _promote_headings(body)
        parts.append(f"\n\n<!-- pagina {i + 1} -->\n\n{body}")
        print(f"  {pdf_path.stem}: pagina {i + 1}/{doc.page_count}   ", end="\r")

    doc.close()
    print()
    return "\n".join(parts)


def _ocr_page(page: pymupdf.Page, dpi: int) -> str:
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    data = pytesseract.image_to_data(
        img[:, :, :3],
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    items = []
    for j, text in enumerate(data["text"]):
        text = text.strip()
        if not text or int(data["conf"][j]) < 30:
            continue
        left = data["left"][j]
        top = data["top"][j]
        w = data["width"][j]
        h = data["height"][j]
        items.append({
            "text": text,
            "left": left,
            "right": left + w,
            "mid_y": top + h / 2,
            "height": h,
        })

    return _reconstruct_page(items, page_width_px=pix.width)


def _reconstruct_page(items: list[dict], page_width_px: int) -> str:
    """
    Recebe palavras com coordenadas. Agrupa palavras com y parecido numa
    linha, ordena da esquerda para a direita, e insere espaçamento
    proporcional ao vão horizontal (para tabelas manterem noção de coluna).
    """
    if not items:
        return ""

    items.sort(key=lambda it: (it["mid_y"], it["left"]))
    median_h = sorted(it["height"] for it in items)[len(items) // 2] or 1

    lines: list[list[dict]] = []
    for it in items:
        if lines and abs(it["mid_y"] - lines[-1][0]["mid_y"]) <= median_h * 0.6:
            lines[-1].append(it)
        else:
            lines.append([it])

    out_lines = []
    for line in lines:
        line.sort(key=lambda it: it["left"])
        buf = line[0]["text"]
        for prev, cur in zip(line, line[1:]):
            gap = cur["left"] - prev["right"]
            # vão > ~3.5% da largura da página => provável separação de coluna
            sep = "    " if gap > page_width_px * 0.035 else " "
            buf += sep + cur["text"]
        buf = buf.rstrip()
        if buf and not any(p.search(buf) for p in JUNK_PATTERNS):
            out_lines.append(buf)

    return "\n".join(out_lines)


def _promote_headings(text: str) -> str:
    """Transforma 'N. Título' em '### N. Título' para o chunking estrutural."""
    out = []
    for ln in text.splitlines():
        m = NOTE_HEADING.match(ln.strip())
        out.append(f"### {ln.strip()}" if m else ln)
    return "\n".join(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="OCR de um PDF -> Markdown (stdout ou arquivo)")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, help="salvar em arquivo em vez de stdout")
    ap.add_argument("--dpi", type=int, default=OCR_DPI)
    args = ap.parse_args()

    md = ocr_pdf_to_markdown(args.pdf, dpi=args.dpi)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"salvo em {args.out}")
    else:
        print(md)
