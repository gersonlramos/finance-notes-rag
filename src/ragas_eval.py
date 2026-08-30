"""
Milestone 5 (parte 2) — RAGAS
=============================

Métricas RAGAS sobre as respostas já geradas (cache em data/eval/runs/).

  faithfulness       : a resposta se sustenta no contexto? (mede alucinação)
  answer_relevancy   : a resposta de fato responde a pergunta feita?
  context_precision  : os chunks recuperados são relevantes ou vem lixo junto?
  context_recall     : os chunks recuperados cobrem tudo que a referência exige?

RAGAS usa um LLM JUIZ. Aqui o juiz é o próprio Ollama local (custo zero), o que
é uma limitação: um modelo 3B julga mal. Os números servem para comparar
configurações entre si, não como verdade absoluta. Para números confiáveis,
troque `config.RAGAS_JUDGE_MODEL` por um modelo forte (ex: via API) — a única
etapa do projeto que não é 100% local.

Chamado por `evaluate.py --ragas`. Lento: cada métrica faz várias chamadas ao
LLM por pergunta.
"""

from __future__ import annotations

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src import config
from src.embed_index import get_embed_model
from src.rag import get_llm

RAGAS_JUDGE_MODEL = getattr(config, "RAGAS_JUDGE_MODEL", config.OLLAMA_MODEL)


def evaluate_with_ragas(items: list[dict], results_by_id: dict, console) -> None:
    samples = []
    for it in items:
        r = results_by_id.get(it["id"])
        if not r or not r.get("contexts"):
            continue
        samples.append(SingleTurnSample(
            user_input=it["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            # referência: a resposta esperada (usada por context_recall e relevancy)
            reference="; ".join(it["expected"]),
        ))

    if not samples:
        raise SystemExit("Sem respostas em cache. Rode `python -m src.evaluate` primeiro.")

    judge = LlamaIndexLLMWrapper(get_llm(RAGAS_JUDGE_MODEL))
    embeds = LlamaIndexEmbeddingsWrapper(get_embed_model())

    console.print(f"\n[bold]RAGAS[/bold] · {len(samples)} perguntas · juiz: {RAGAS_JUDGE_MODEL} "
                  f"[dim](local, lento — pode levar bastante)[/dim]")

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=embeds),
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge),
        ],
        show_progress=True,
    )

    console.print("\n[bold]Resultado RAGAS (média)[/bold]")
    for metric, value in result._repr_dict.items():
        console.print(f"  {metric:24} {value:.3f}")
    df = result.to_pandas()
    df.to_csv(config.DATA_DIR / "eval" / "ragas_scores.csv", index=False)
    console.print(f"[dim]por pergunta em data/eval/ragas_scores.csv[/dim]")
