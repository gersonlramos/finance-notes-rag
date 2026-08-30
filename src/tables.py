"""
Milestone 6 — Linearização de tabelas
=====================================

A avaliação (docs/avaliacao.md) mostrou que o gargalo é o LLM 3B lendo tabela:
com o chunk certo no contexto, ele ainda pega a sub-linha em vez do total, ou a
coluna "consolidado" em vez de "controladora".

Solução: antes de indexar, converter cada LINHA de tabela numa FRASE
autocontida:

    |Total do ativo||1.389.902|1.131.893|
    ->
    Balanço patrimonial — Total do ativo em 2023: 1.389.902 mil reais.
    Balanço patrimonial — Total do ativo em 2022: 1.131.893 mil reais.

A frase é fácil para o BM25 (casa "total do ativo" + "2023") e trivial para o
LLM (é só ler). A tabela original é mantida; as frases entram DEPOIS dela, num
bloco "Fatos:".

Trata os dois formatos que aparecem nos documentos:
  - Markdown  (|a|b|c|)          -> docs de 2022, 2023, 2026
  - OCR       (colunas por 2+ espaços) -> docs de 2024, 2025
"""

from __future__ import annotations

import re

YEAR = re.compile(r"\b(20[12]\d)\b")
DATE_OR_YEAR = re.compile(r"^(?:\d{1,2}/\d{1,2}/)?20[12]\d$")
NUM = re.compile(r"^\(?-?[\d.]+\)?$")          # 1.389.902  (734)  -
HEADING = re.compile(r"^#{1,6}\s+(.*\S)")
UNIT_HINT = re.compile(r"milhares|mil reais", re.I)

# rótulos que são só seção, não linha de dado
SECTION_LABELS = {
    "ativo", "ativos", "passivo", "passivos", "circulante", "não circulante",
    "nao circulante", "patrimônio líquido", "patrimonio liquido",
}


def _is_num(cell: str) -> bool:
    return bool(NUM.match(cell.strip())) and any(c.isdigit() for c in cell)


def _md_cells(row: str) -> list[str]:
    """'||Total|3|10|' -> ['', 'Total', '3', '10'] (remove só os delimitadores das pontas)."""
    parts = row.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [c.strip() for c in parts]


def _sentences_for_row(label: str, note: str | None, pairs: list[tuple[str, str]],
                       context: str, unit: str) -> list[str]:
    """pairs = [(descrição da coluna, valor), ...]."""
    label = label.strip(" |")
    if not label or label.lower() in SECTION_LABELS:
        return []
    out = []
    ctx = f"{context} — " if context else ""
    nota = f" (Nota {note})" if note else ""
    for coldesc, value in pairs:
        value = value.strip()
        if not value or value in {"-", "–"} or not any(c.isdigit() for c in value):
            continue
        out.append(f"{ctx}{label}{nota}, {coldesc}: {value}{unit}.")
    return out


def _scope_map(year_idx: list[int], header: list[str], supers: list[str]) -> dict[int, str | None]:
    """
    Associa cada coluna de ano a um super-cabeçalho (Controladora/Consolidado).
    Um ano que se repete marca o início de um novo grupo:
    '2024 2023 2024' -> [Controladora: 2024, 2023] [Consolidado: 2024].
    """
    if len(supers) < 2 or len(year_idx) < 2:
        return {ci: None for ci in year_idx}
    out, g, seen = {}, 0, set()
    for ci in year_idx:
        key = header[ci].strip()
        if key in seen:
            g = min(g + 1, len(supers) - 1)
            seen = set()
        out[ci] = supers[g]
        seen.add(key)
    return out


def _col_descs_from_years(header_cells: list[str]) -> list[str | None]:
    """Cada célula de cabeçalho vira 'em 20xx' ou None (coluna 'Nota')."""
    descs = []
    for c in header_cells:
        c = c.strip()
        if DATE_OR_YEAR.match(c):
            descs.append(f"em {c}")
        elif c.lower() in {"nota", "notas"}:
            descs.append(None)          # marcador: é a coluna de nota
        elif c == "":
            descs.append(None)
        else:
            descs.append(c or None)
    return descs


# ---------------------------------------------------------------- Markdown
def _linearize_md_table(lines: list[str], context: str, unit: str) -> list[str]:
    rows = [l for l in lines if l.strip().startswith("|") and set(l.strip()) - {" "} != {"|", "-"}]
    if len(rows) < 2:
        return []
    # o cabeçalho é a 1ª linha que tem ano/data (pula super-cabeçalhos quebrados)
    hi = next((i for i, r in enumerate(rows) if DATE_OR_YEAR.search(
        " ".join(_md_cells(r)).replace(" ", ""))
        or any(DATE_OR_YEAR.match(c) for c in _md_cells(r))), None)
    if hi is None:
        return []
    header = _md_cells(rows[hi])
    descs = _col_descs_from_years(header)
    year_idx = [i for i, d in enumerate(descs) if d and d.startswith("em ")]
    if not year_idx:
        return []                       # sem coluna de ano -> não linearizamos

    # super-cabeçalho (Controladora/Consolidado) numa linha | anterior ao header
    supers = []
    for r in rows[:hi]:
        cs = [c for c in _md_cells(r) if c and not c.isdigit()]
        joined = " ".join(cs).replace("Contro ladora", "Controladora").replace("Conso lidado", "Consolidado")
        supers = [w for w in joined.split() if w.lower() in {"controladora", "consolidado"}]
        if supers:
            break
    scope = _scope_map(year_idx, header, supers)

    facts = []
    for row in rows[hi + 1:]:
        cells = _md_cells(row)
        if len(cells) != len(header):
            continue
        label = cells[0]
        note = next((cells[i] for i, d in enumerate(descs)
                     if d is None and i > 0 and cells[i] and cells[i] not in {"-", ""}
                     and re.fullmatch(r"\d{1,2}(\.\d+)?", cells[i])), None)
        pairs = []
        for i in year_idx:
            val = cells[i]
            sc = scope.get(i)
            col = f"{sc.lower()}, {descs[i]}" if sc else descs[i]
            pairs.append((col, val))
        facts += _sentences_for_row(label, note, pairs, context, unit)
    return facts


# ---------------------------------------------------------------- OCR
def _split_ocr(line: str) -> list[str]:
    return [c for c in re.split(r"\s{2,}", line.strip()) if c != ""]


def _linearize_ocr_table(header_line: str, super_line: str | None,
                         data_lines: list[str], context: str, unit: str) -> list[str]:
    header = _split_ocr(header_line)          # ex: ["Nota","2024","2023","2024"]
    supers = _split_ocr(super_line) if super_line else []   # ex: ["Controladora","Consolidado"]
    descs = _col_descs_from_years(header)
    year_idx = [i for i, d in enumerate(descs) if d and d.startswith("em ")]
    if not year_idx:
        return []

    scope_for = _scope_map(year_idx, header, supers)

    n_years = len(year_idx)
    yr_descs = [descs[i] for i in year_idx]
    yr_scopes = [scope_for.get(i) for i in year_idx]
    has_note_col = descs and descs[0] is None and header[0].strip().lower().startswith("nota")

    facts, parsed = [], 0
    for line in data_lines:
        cells = _split_ocr(line)
        # Conservador: as N últimas células têm que ser TODAS numéricas, e sobrar
        # exatamente 1 (rótulo) ou 2 (rótulo + nota) células antes delas.
        if len(cells) < n_years + 1:
            continue
        values = cells[-n_years:]
        if not all(_is_num(v) for v in values):
            continue
        head = cells[: len(cells) - n_years]
        note = None
        if has_note_col and len(head) >= 2 and re.fullmatch(r"\d{1,2}(\.\d+)?", head[-1]):
            note, head = head[-1], head[:-1]
        label = " ".join(head).strip()
        # rótulo com dígito ou marcador de nota de rodapé -> parsing duvidoso, pula
        if re.search(r"\d", label) or re.search(r"\((?:i{1,3}|[a-e])\)", label):
            continue

        pairs = [(f"{sc.lower()}, {d}" if sc else d, v)
                 for d, sc, v in zip(yr_descs, yr_scopes, values)]
        rowfacts = _sentences_for_row(label, note, pairs, context, unit)
        facts += rowfacts
        if rowfacts:
            parsed += 1

    # se quase nada parseou, não entendemos a tabela -> não emite nada
    if parsed < max(2, len(data_lines) // 3):
        return []
    return facts


# ---------------------------------------------------------------- driver
def linearize(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    context = ""
    unit = ""
    i = 0
    while i < len(lines):
        line = lines[i]

        h = HEADING.match(line)
        if h:
            context = h.group(1)
        if UNIT_HINT.search(line):
            unit = " mil reais" if "milhares" in line.lower() or "mil reais" in line.lower() else unit
        # o título da tabela não vira heading (não tem #), mas é bom contexto:
        if re.search(r"balan[çc]o patrimonial|demonstra[çc][ãa]o d[oa]|receita operacional", line, re.I):
            context = re.sub(r"\s+31 de dezembro.*|\s+\(Em milhares.*", "", line).strip(" #*")

        # bloco de tabela markdown
        if line.strip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            block = lines[i:j]
            out += block
            facts = _linearize_md_table(block, context, unit or " mil reais")
            if facts:
                out.append("")
                out.append("Fatos: " + " ".join(facts))
            i = j
            continue

        # bloco de tabela OCR: linha de cabeçalho com "Nota  20xx" ou "20xx  20xx"
        if re.match(r"^\s*(Nota\s+)?20[12]\d(\s+20[12]\d)+\s*$", line) or \
           re.match(r"^\s*Nota\s+20[12]\d", line):
            super_line = lines[i - 1] if i > 0 and re.match(
                r"^\s*(Controladora|Consolidado)", lines[i - 1]) else None
            j = i + 1
            data = []
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith("#") or s.startswith("|"):
                    break
                if re.search(r"\d", s) and re.search(r"\s{2,}", s):
                    data.append(lines[j])
                    j += 1
                elif not re.search(r"\d", s) and len(s.split()) <= 4:
                    data.append(lines[j])   # linha de subtítulo dentro da tabela
                    j += 1
                else:
                    break
            out.append(line)
            out += data
            facts = _linearize_ocr_table(line, super_line, data, context, unit or " mil reais")
            if facts:
                out.append("")
                out.append("Fatos: " + " ".join(facts))
            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)
