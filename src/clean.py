"""
Milestone 1.5 — Limpeza do texto extraído
=========================================

Fica ENTRE a extração (Etapa 1) e o chunking (Etapa 2):

    PDF -> data/extracted/<doc>.md  --[clean]-->  data/clean/<doc>.md  -> chunks

Por que existe: a extração de PDF de design (colunas, texto colorido,
capitulares "esticadas") deixa lixo que envenena os embeddings e fragmenta o
chunking. Cada função aqui resolve UM tipo de lixo, e o `__main__` mostra o
antes/depois de cada regra para você ver o efeito.

Regras (na ordem em que rodam):
  1. strip_html_tags      — <br> <u> <mark> <sub> ... viram espaço/nada
  2. strip_emphasis       — remove **negrito**, __ , ~~riscado~~ (mantém o texto)
  3. drop_garbled_lines   — descarta linhas decorativas ("R E L A T Ó R I O ...")
  4. collapse_spacing     — conserta espaçamento residual ("2 0 2 3" -> "2023")
  5. drop_boilerplate     — "Docusign Envelope ID: ...", número de página solto
  6. drop_running_headers — cabeçalho/rodapé de página (lista explícita, para
                            NÃO remover cabeçalho de coluna de tabela nem a
                            anotação de unidade "em milhares de reais")
  7. squeeze_blank_lines  — no máximo uma linha em branco entre blocos
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLEAN_DIR = config.DATA_DIR / "clean"

_HTML_TAG = re.compile(r"</?(?:br|u|mark|sub|sup|span|div|i|b|em|strong)\b[^>]*>", re.I)
_EMPHASIS = re.compile(r"(\*\*|__|~~)")
_BOILERPLATE = [
    re.compile(r"^\s*docusign envelope id:.*$", re.I),
    re.compile(r"^\s*\d{1,3}\s*$"),                 # número de página sozinho
    re.compile(r"^\s*<!--.*-->\s*$"),               # comentários (marcadores de página do OCR)
    re.compile(r"^\s*#{1,6}\s*$"),                  # heading vazio ("######")
]
# Espaçamento residual: 4+ caracteres isolados separados por 1 espaço.
_SPACED_RUN = re.compile(r"(?:(?<=\s)|^)((?:[0-9A-Za-zÀ-ÿ] ){3,}[0-9A-Za-zÀ-ÿ])(?=\s|$)")

# Cabeçalho/rodapé de página — lista EXPLÍCITA. Deliberadamente conservadora:
# melhor deixar passar um pouco de ruído do que apagar cabeçalho de coluna de
# tabela ("Controladora  Consolidado", "2025  2024") ou a unidade dos valores.
_RUNNING_HEADER = [
    re.compile(r"^#{0,6}\s*clube de regatas do flamengo\s*$", re.I),
    re.compile(r"^#{0,6}\s*relat[óo]rio\s*$", re.I),
    re.compile(r"^de gest[ãa]o\s*$", re.I),
    re.compile(r"^demonstra[çc][õo]es financeiras com\s*$", re.I),
    re.compile(r"^(parecer d[eo] )?auditor(ia)? independente\s*$", re.I),
    re.compile(r"^parecer\s*$", re.I),
    re.compile(r"^relat[óo]rio\s+de transpar[êe]ncia econ[ôo]mico-financeiro.*$", re.I),
    re.compile(r"^\d{1,2}\s+de\s+\w+\s+de\s+20\d{2}\s*$", re.I),  # "31 de dezembro de 2025"
    re.compile(r"^flamengo\s*$", re.I),
]


def strip_control_chars(text: str) -> str:
    """Remove caracteres de controle (lixo de OCR de capas estilizadas)."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def strip_html_tags(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    return _HTML_TAG.sub("", text)


def strip_emphasis(text: str) -> str:
    return _EMPHASIS.sub("", text)


def _is_garbled(line: str) -> bool:
    """
    Linha decorativa tipo 'R E L A T Ó R I O A N U A L': muitas LETRAS isoladas.
    Cuidado para NÃO pegar linha de tabela ('Outros resultados  -  -  -'), que
    tem colunas separadas por 2+ espaços e traços/números como células vazias.
    """
    s = line.strip()
    if "|" in s or re.search(r"\S {2,}\S", s):   # é tabela -> não é decorativa
        return False
    toks = s.split()
    if len(toks) < 6:
        return False
    single_letters = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    return single_letters / len(toks) >= 0.5


def drop_garbled_lines(text: str) -> str:
    return "\n".join("" if _is_garbled(ln) else ln for ln in text.splitlines())


def collapse_spacing(text: str) -> str:
    """
    'D E Z E M B R O' -> 'DEZEMBRO', '2 0 2 3' -> '2023'.

    Limitação conhecida: um título decorativo TODO espaçado com uma só folga
    entre palavras ('C A I X A E E Q U I V A L E N T E S') vira uma palavra
    colada ('CAIXAEEQUIVALENTES'). Sem dicionário não dá para saber onde a
    palavra termina. Afeta só alguns títulos dos relatórios de gestão 2022/2023;
    as demonstrações contábeis em si não têm esse problema.
    """
    def _join(m: re.Match) -> str:
        return m.group(1).replace(" ", "")

    return "\n".join(_SPACED_RUN.sub(_join, ln) for ln in text.splitlines())


def drop_boilerplate(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if any(p.match(ln) for p in _BOILERPLATE):
            continue
        out.append(ln)
    return "\n".join(out)


def drop_running_headers(text: str) -> str:
    """
    Remove linhas de cabeçalho/rodapé de página, identificadas por padrão
    explícito (_RUNNING_HEADER). NÃO remove por frequência — cabeçalho de
    coluna de tabela também repete e precisa ficar.
    """
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and any(p.match(s) for p in _RUNNING_HEADER):
            continue
        out.append(ln)
    return "\n".join(out)


def squeeze_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


_PIPELINE = [
    strip_control_chars,
    strip_html_tags,
    strip_emphasis,
    drop_garbled_lines,
    collapse_spacing,
    drop_boilerplate,
    drop_running_headers,
    squeeze_blank_lines,
]


def clean_markdown(text: str) -> str:
    for step in _PIPELINE:
        text = step(text)
    return text


# ======================================================================
def _diff_stats(before: str, after: str) -> str:
    b_lines = [l for l in before.splitlines() if l.strip()]
    a_lines = [l for l in after.splitlines() if l.strip()]
    return (f"linhas {len(b_lines)} -> {len(a_lines)} "
            f"({len(b_lines) - len(a_lines):+d}) | "
            f"chars {len(before)} -> {len(after)} ({len(after) - len(before):+d})")


def _run_all() -> None:
    src_files = sorted(config.EXTRACTED_DIR.glob("*.md"))
    if not src_files:
        raise SystemExit(f"Nada em {config.EXTRACTED_DIR}. Rode o ingest primeiro.")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for path in src_files:
        raw = path.read_text(encoding="utf-8")
        print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")

        text = raw
        for step in _PIPELINE:
            before = text
            text = step(text)
            if before == text:
                continue
            print(f"  {step.__name__:22} {_diff_stats(before, text)}")

            b_lines = [l.strip() for l in before.splitlines() if l.strip()]
            a_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(b_lines) > len(a_lines):        # regra APAGA linhas
                a_set = set(a_lines)
                removed = [l for l in b_lines if l not in a_set]
                print(f"  {'':22} removeu: {' | '.join(dict.fromkeys(removed))[:200]}")
            else:                                  # regra TRANSFORMA linhas no lugar
                pairs = [(x, y) for x, y in zip(b_lines, a_lines) if x != y][:2]
                for x, y in pairs:
                    print(f"  {'':22} {x[:66]!r} -> {y[:66]!r}")

        out = CLEAN_DIR / path.name
        out.write_text(text, encoding="utf-8")
        print(f"  -> {out}   (total: {_diff_stats(raw, text)})")


if __name__ == "__main__":
    _run_all()
