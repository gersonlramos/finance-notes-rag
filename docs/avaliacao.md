# Avaliação do pipeline

Conjunto de teste: 20 perguntas em `data/eval/questions.jsonl` — 17 com resposta
nos documentos (categorias: balanço, resultado, nota explicativa, narrativa) e
3 sem resposta (para testar se o sistema admite não saber). Gabarito verificado
manualmente nas demonstrações.

## Métricas (determinísticas, sem LLM juiz)

| Métrica | O que mede |
|---|---|
| `retrieval hit@k` | o valor/termo esperado apareceu em algum dos `k` chunks recuperados |
| `MRR` | 1 / (posição do primeiro chunk com o valor esperado) |
| `resposta correta` | a resposta gerada contém o valor/termo esperado |
| `abstenção correta` | nas perguntas sem resposta, o modelo respondeu "não encontrei" |

Ambiente: 100% local, CPU (i5-8265U). Embeddings `paraphrase-multilingual-MiniLM-L12-v2`
(ONNX), LLM `llama3.2:3b` via Ollama, chunking estrutural, `k=3`.

## Comparação de configurações de retrieval

| Configuração | retrieval hit@k | MRR | resposta correta | abstenção |
|---|---|---|---|---|
| `hybrid`, sem filtro de ano | 71% | 0,55 | 35% (6/17) | 100% |
| `hybrid` + filtro de ano | 59% | 0,41 | 24% (4/17) | 100% |
| `bm25` + filtro de ano (oráculo) | 88% | 0,71 | 65% (11/17) | 100% |
| `bm25` + ano inferido, chunking estrutural | 88% | 0,67 | 59% (10/17) | 100% |
| **`bm25` + ano inferido, chunking fixo** | 82% | **0,75** | **65% (11/17)** | **100%** |

### Leitura dos resultados

1. **A busca semântica atrapalha nesta base.** O modelo de embedding multilíngue
   pequeno (384 dims) é grosseiro demais para a terminologia contábil exata. A
   fusão `hybrid` (RRF) mistura o ranking bom do BM25 com o ruído do vetorial e
   piora o resultado. BM25 puro + filtro de exercício quase dobra a acurácia.

2. **Inferir o ano da pergunta funciona.** Extrair "2024" do texto da pergunta e
   filtrar por `doc_year` recupera 88% — mesmo patamar do filtro manual — sem o
   usuário precisar especificar.

3. **O gargalo restante é a leitura de tabela pelo LLM 3B**, não o retrieval.
   Com 88% de retrieval hit, a acurácia da resposta é 59%: quando o chunk certo
   está no contexto, o modelo ainda erra ~1/3 das vezes — pega a sub-linha em
   vez do total, ou a coluna "consolidado" em vez de "controladora".

4. **Abstenção perfeita (100%).** O sistema nunca inventou resposta para
   pergunta sem base documental. O prompt de sistema ("diga *não encontrei*")
   segura isso mesmo num modelo pequeno.

## Experimento: modelo de geração

Mesma config (chunking fixo + `bm25` + ano inferido), variando o LLM do Ollama:

| Modelo | resposta correta | abstenção | tempo/pergunta |
|---|---|---|---|
| `gemma2:2b` | 59% | **falhou** (inventou salário) | 55 s |
| **`llama3.2:3b`** (padrão) | 65% | 100% | 64 s |
| `qwen2.5:7b-instruct` | **71%** | 100% | 153 s |

- **`gemma2:2b`** é ~15% mais rápido mas quebra a abstenção — inventou "salário
  do técnico R$ 9.756.000" para uma pergunta sem resposta. Descartado.
- **`qwen2.5:7b`** acerta +6 pontos e mantém a abstenção, ao custo de 2,4× o
  tempo (~2,5 min/resposta). Fica disponível como opção no seletor da interface.
- A categoria `balanco` **não melhorou** com o 7B — lá a falha é o retrieval
  (o chunk certo entra no top-3 só 40% das vezes), não o modelo.
- Nota: modelos ≥ 7B precisam de `num_gpu=0` nesta máquina — a MX110 (2 GB)
  não cabe as camadas e o offload deixa ~5× mais lento.

`llama3.2:3b` continua o padrão pelo equilíbrio velocidade / qualidade / segurança.

## Experimento: chunking estrutural vs. tamanho fixo

Mesma config (`bm25` + ano inferido, `llama3.2:3b`), variando a estratégia de
chunking:

| | estrutural (por título) | tamanho fixo (512/64) |
|---|---|---|
| nº de chunks | 504 | 514 |
| mediana de tokens | 309 | 449 |
| retrieval hit@k | **88%** | 82% |
| MRR | 0,67 | **0,75** |
| **resposta correta** | 59% | **65%** |
| tempo/pergunta | 52 s | 60 s |

**O chunking fixo venceu na métrica que importa (+6 pontos na resposta)**, apesar
de recuperar um pouco menos. Contraintuitivo: no Milestone 1 supus que o chunking
estrutural seria melhor por manter cada tabela inteira num chunk. A avaliação
mostrou o contrário — os chunks estruturais são muito desiguais (tabelas de
1000+ tokens ao lado de fragmentos de título), e o modelo pequeno lida melhor
com blocos de tamanho uniforme. **Medir, não supor.**

Configuração padrão do projeto passou a ser: chunking fixo + `bm25` + ano inferido.

## Acurácia por categoria (`bm25` + ano inferido, chunking estrutural)

| Categoria | n | retrieval | resposta ok |
|---|---|---|---|
| balanço | 5 | 60% | 40% |
| resultado (DRE) | 4 | 100% | 75% |
| nota explicativa | 5 | 100% | 60% |
| narrativa | 3 | 100% | 67% |
| sem resposta | 3 | — | 100% |

Balanço é o ponto fraco: "total do ativo" é uma expressão genérica (BM25 tem
mais dificuldade) e o valor fica enterrado numa tabela markdown de ~1000 tokens.

## Limitação: RAGAS

O plano previa RAGAS (faithfulness, answer relevancy, context precision/recall).
Duas barreiras práticas:

- **Dependências:** RAGAS exige `langchain 0.2.x`, que fixa `numpy<2` e conflita
  com o resto do stack. Roda só em ambiente isolado (`src/ragas_eval.py`).
- **Juiz:** RAGAS precisa de um LLM juiz capaz. O único disponível localmente é
  o mesmo `llama3.2:3b` — que já vimos ser fraco em leitura de tabela, logo um
  juiz ruim. Os números seriam pouco confiáveis.

Para Q&A financeiro, onde a resposta é um número, o *matching* determinístico
(acima) é mais direto e reproduzível. RAGAS entraria como complemento com um
juiz forte (via API) — a única etapa não-local do projeto.

## Experimento: linearização de tabelas (resultado negativo)

Hipótese: converter cada linha de tabela numa frase autocontida
(`Balanço patrimonial — Total do ativo, em 2023: 1.389.902 mil reais`) tornaria
o valor mais fácil de recuperar e de ler. Implementado em `src/tables.py`
(flag `python -m src.ingest --linearize`).

| Métrica | baseline | + linearização |
|---|---|---|
| retrieval hit@k | 88% | 82% |
| resposta correta | **59%** | **29%** |
| categoria narrativa | 67% | 0% |

**Piorou 30 pontos.** Causas:

1. **Diluição do índice** — ~centenas de frases quase idênticas
   (`... X, em YYYY: N`). O BM25 passa a ter muitos candidatos parecidos e erra
   a escolha (respondeu a coluna de 2022 em vez de 2023; "Total do passivo" em
   vez de "Total do ativo").
2. **Narrativa soterrada** — os chunks de prosa deixaram de ser recuperados.
3. **Geração pior** — contexto cheio de frases curtas e parecidas confunde mais
   o modelo pequeno.

Lição: adicionar texto estruturado sem (a) remover o original, (b) isolar num
índice próprio ou (c) ajustar o retrieval, inunda o índice. Uma abordagem
melhor: índice de fatos separado, consultado só para perguntas numéricas.

## Próximos passos

- Índice de fatos separado (não misturado com a prosa).
- Modelo de geração maior (a leitura de tabela é o gargalo, não o retrieval).
- OCR de tabela com reconstrução de estrutura (2024/2025).
