"""
Milestone 5 — Avaliação
=======================

Cobre a **Etapa 7**. Rodamos o pipeline sobre um conjunto de perguntas de teste
(`data/eval/questions.jsonl`) e medimos a qualidade — com números, não "achismo".

Duas camadas de métrica:

1. DETERMINÍSTICAS (sem LLM, grátis, rodam sempre)
   - retrieval_hit@k : o valor esperado apareceu em algum chunk recuperado?
   - MRR             : 1 / (posição do primeiro chunk com o valor esperado)
   - answer_correct  : a resposta contém o valor/termo esperado?
   - abstained_ok    : nas perguntas sem resposta, o modelo disse "não encontrei"?

2. RAGAS (opcional, --ragas) — faithfulness, answer_relevancy,
   context_precision, context_recall. Precisa de um LLM juiz (ver --help).

Uso:
    python -m src.evaluate                      # roda o pipeline + métricas determinísticas
    python -m src.evaluate --limit 5            # só as 5 primeiras (teste rápido)
    python -m src.evaluate --mode bm25          # avalia outro modo de retrieval
    python -m src.evaluate --report-only        # só recalcula o relatório do cache
    python -m src.evaluate --ragas              # + métricas RAGAS sobre o cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import config, rag

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

QUESTIONS = config.DATA_DIR / "eval" / "questions.jsonl"
RUNS_DIR = config.DATA_DIR / "eval" / "runs"


# ---------------------------------------------------------------- matching
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _matches(text: str, expected: str) -> bool:
    """
    `expected` pode ser um número ("1.389.902") ou um termo ("Copa do Brasil").
    Número: casa com separadores flexíveis e sem colar em outro dígito
            (734 não casa dentro de 1734).
    Termo:  substring, sem acento, sem caixa.
    """
    if re.fullmatch(r"[\d.\s]+", expected):
        digits = re.sub(r"\D", "", expected)
        pat = r"(?<!\d)" + r"[.\s]?".join(digits) + r"(?!\d)"
        return re.search(pat, text) is not None
    return _strip_accents(expected).lower() in _strip_accents(text).lower()


def _any_match(text: str, expected_list: list[str]) -> bool:
    return any(_matches(text, e) for e in expected_list)


# ---------------------------------------------------------------- run pipeline
def _run_key(strategy: str, mode: str, k: int, model: str) -> str:
    raw = f"{strategy}|{mode}|{k}|{model}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def run_pipeline(items: list[dict], *, strategy: str, mode: str, k: int, model: str,
                 use_year_oracle: bool) -> list[dict]:
    """Roda o RAG em cada pergunta, com cache em data/eval/runs/."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RUNS_DIR / f"{_run_key(strategy, mode, k, model)}{'_yr' if use_year_oracle else ''}.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            cache[o["id"]] = o

    results = []
    for it in items:
        if it["id"] in cache:
            results.append(cache[it["id"]])
            continue
        year = it["gold_year"] if use_year_oracle else None
        console.print(f"[dim]{it['id']}: {it['question'][:60]}...[/dim]")
        r = rag.run(it["question"], strategy=strategy, mode=mode, k=k, year=year, model=model)
        rec = {
            "id": it["id"],
            "question": it["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "seconds": round(r["seconds"], 1),
        }
        with cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results.append(rec)
    return results, cache_path


# ---------------------------------------------------------------- scoring
def score(items: list[dict], results: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in results}
    rows = []
    for it in items:
        r = by_id[it["id"]]
        exp = it["expected"]
        answerable = it["category"] != "sem_resposta"

        hit_rank = None
        for i, ctx in enumerate(r["contexts"]):
            if _any_match(ctx, exp):
                hit_rank = i + 1
                break

        answer_low = _strip_accents(r["answer"]).lower()
        if answerable:
            answer_ok = _any_match(r["answer"], exp)
        else:
            answer_ok = "nao encontrei" in answer_low

        rows.append({
            "id": it["id"],
            "category": it["category"],
            "answerable": answerable,
            "retrieval_hit": hit_rank is not None if answerable else None,
            "hit_rank": hit_rank,
            "mrr": (1.0 / hit_rank) if hit_rank else (0.0 if answerable else None),
            "answer_ok": answer_ok,
            "answer": r["answer"],
            "seconds": r.get("seconds", 0.0),
        })
    return rows


def report(rows: list[dict]) -> None:
    ans = [r for r in rows if r["answerable"]]
    noans = [r for r in rows if not r["answerable"]]

    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"

    console.print("\n[bold]Agregado[/bold]")
    console.print(f"  retrieval hit@k (respondíveis):  {pct([r['retrieval_hit'] for r in ans])}"
                  f"   ({sum(r['retrieval_hit'] for r in ans)}/{len(ans)})")
    console.print(f"  MRR médio:                       {sum(r['mrr'] for r in ans) / len(ans):.2f}")
    console.print(f"  resposta correta (respondíveis): {pct([r['answer_ok'] for r in ans])}"
                  f"   ({sum(r['answer_ok'] for r in ans)}/{len(ans)})")
    console.print(f"  abstenção correta (sem resposta):{pct([r['answer_ok'] for r in noans])}"
                  f"   ({sum(r['answer_ok'] for r in noans)}/{len(noans)})")
    console.print(f"  tempo médio/pergunta:            {sum(r['seconds'] for r in rows) / len(rows):.0f}s")

    cats = sorted({r["category"] for r in rows})
    t = Table(title="Por categoria")
    for c in ("categoria", "n", "retr. hit", "resp. ok"):
        t.add_column(c)
    for c in cats:
        g = [r for r in rows if r["category"] == c]
        rh = [r["retrieval_hit"] for r in g if r["retrieval_hit"] is not None]
        t.add_row(c, str(len(g)), pct(rh), pct([r["answer_ok"] for r in g]))
    console.print(t)

    fails = [r for r in rows if not r["answer_ok"]]
    if fails:
        console.print("\n[bold red]Falhas[/bold red]")
        for r in fails:
            flag = "" if r["retrieval_hit"] or not r["answerable"] else "  [chunk certo NÃO recuperado]"
            console.print(f"  [yellow]{r['id']}[/yellow] ({r['category']}){flag}")
            console.print(f"    resposta: {r['answer'][:160]}")


# ---------------------------------------------------------------- RAGAS
def run_ragas(items: list[dict], results: list[dict], n: int) -> None:
    try:
        from src.ragas_eval import evaluate_with_ragas
    except ImportError as e:
        raise SystemExit(
            f"RAGAS não instalado neste ambiente ({e}).\n"
            "O RAGAS exige langchain 0.2.x, que fixa numpy<2 e conflita com o "
            "resto do pipeline. Rode-o num venv separado — ver src/ragas_eval.py."
        )
    evaluate_with_ragas(items[:n], {r["id"]: r for r in results}, console)


# ---------------------------------------------------------------- main
def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação do pipeline (Milestone 5)")
    ap.add_argument("--strategy", choices=["fixed", "structural"], default="fixed")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=config.GENERATION_TOP_K)
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--year-oracle", action="store_true",
                    help="passa o ano correto como filtro (mede o teto do retrieval)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report-only", action="store_true", help="não roda o LLM, só recalcula do cache")
    ap.add_argument("--ragas", action="store_true", help="também roda métricas RAGAS")
    ap.add_argument("--ragas-n", type=int, default=8)
    args = ap.parse_args()

    items = load_questions()
    if args.limit:
        items = items[:args.limit]

    console.print(f"[bold]{len(items)} perguntas[/bold] · {args.strategy}/{args.mode} "
                  f"· k={args.k} · {args.model}"
                  + (" · year-oracle" if args.year_oracle else ""))

    if args.report_only:
        key = _run_key(args.strategy, args.mode, args.k, args.model)
        path = RUNS_DIR / f"{key}{'_yr' if args.year_oracle else ''}.jsonl"
        results = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    else:
        results, path = run_pipeline(items, strategy=args.strategy, mode=args.mode,
                                     k=args.k, model=args.model,
                                     use_year_oracle=args.year_oracle)
        console.print(f"[dim]respostas em {path}[/dim]")

    report(score(items, results))

    if args.ragas:
        run_ragas(items, results, args.ragas_n)


if __name__ == "__main__":
    main()
