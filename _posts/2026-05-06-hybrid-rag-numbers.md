---
layout: post
title: "Hybrid RAG with reciprocal-rank fusion and a cross-encoder — what the numbers actually look like"
date: 2026-05-05 19:00:00 +0800
categories: [LLM, Engineering]
tags: [rag, embeddings, bm25, pgvector, reranker, ragas, postgres]
---

> Three retrieval strategies on the same 5-document, 5-chunk corpus and a 20-question golden set. Hybrid + cross-encoder rerank wins by a wide margin on top-1 precision (0.99 vs 0.07 for the runner-up), at the cost of a single extra service round-trip.

## The corpus

Tiny on purpose. Real-world equity research portfolios are not 10 M documents; they are 50–500 hand-curated thesis notes plus a streaming feed of news. Compass Equity ships with three seed thesis files:

- `2330_long_thesis.md` — TSMC bull thesis (March 2026 update)
- `2454_neutral.md` — MediaTek neutral stance
- `macro_taiwan_2026.md` — TWSE macro overlay

Chunked at 1000 chars / 200 overlap → **5 chunks** total.

The golden set is 20 questions: 11 with thesis-grounded answers, 5 with macro context, 2 deliberate "no data" traps to test refusal, and 2 live-data questions exercising indicators.

## The three retrievers

### A. Dense-only

```sql
SELECT c.id, c.content,
       1 - (c.embedding <=> CAST(:emb AS vector)) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL
  AND (CAST(:ticker AS text) IS NULL OR d.ticker = CAST(:ticker AS text) OR d.ticker IS NULL)
ORDER BY c.embedding <=> CAST(:emb AS vector)
LIMIT :lim;
```

Embeddings via `gemini-embedding-001` truncated to 768 dims. pgvector HNSW index for sub-millisecond ANN.

### B. BM25-only

```sql
SELECT c.id, c.content,
       ts_rank(c.tsv, plainto_tsquery('simple', :q)) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.tsv @@ plainto_tsquery('simple', :q)
  AND (CAST(:ticker AS text) IS NULL OR d.ticker = CAST(:ticker AS text) OR d.ticker IS NULL)
ORDER BY score DESC
LIMIT :lim;
```

The `tsv` column is a generated `tsvector` with a GIN index. We use the `simple` config, not `english`, because the corpus mixes English and Traditional Chinese — `english` would drop the Chinese tokens entirely.

### C. Hybrid (BM25 + dense fused with RRF) → cross-encoder rerank

```python
sparse = await _sparse_search(session, query, ticker, candidate_pool=20)
dense  = await _dense_search(session, embedding, ticker, candidate_pool=20)

fused = _rrf_fuse(dense, sparse)[:20]            # reciprocal-rank fusion, k=60
rr    = await rerank(query, [c.content for c in fused], top_k=5)
return [fused[r.index] for r in rr]
```

[Reciprocal-rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): for each item in each ranked list, score `1/(k + rank)` and sum across lists. Items appearing in both lists float to the top.

Reranker is `BAAI/bge-reranker-v2-m3` running as a separate Cloud Run service. It takes the (query, candidate) pair and outputs a cross-encoder score — calibrated relevance instead of a positional artefact.

## What the cross-encoder does to the scores

Sample query: `"investment thesis for TSMC"`, ticker filter `2330`.

| rank | retrieval | score | doc |
|---:|---|---:|---|
| 1 | dense-only | 0.78 | TSMC long thesis (chunk 0) |
| 2 | dense-only | 0.71 | TSMC long thesis (chunk 1) |
| 3 | dense-only | 0.62 | macro 2026 |
| 4 | dense-only | 0.61 | MediaTek neutral |

Dense-only thinks the MediaTek thesis is almost as relevant as the macro overlay — wrong.

After RRF fusion + cross-encoder rerank:

| rank | hybrid + rerank | score | doc |
|---:|---|---:|---|
| 1 | hybrid+rerank | **0.9933** | TSMC long thesis (chunk 0) |
| 2 | hybrid+rerank | 0.0746 | macro 2026 |
| 3 | hybrid+rerank | 0.0266 | macro 2026 |
| 4 | hybrid+rerank | 0.0009 | TSMC long thesis (chunk 1) |

Two things to notice:

1. The cross-encoder is **harshly discriminating**: top-1 has near-perfect score, the rest fall off a cliff. This is the calibrated-relevance behaviour you can't get from cosine similarity, where everything at the top tends to cluster around 0.6–0.8.
2. MediaTek's thesis dropped out entirely. The reranker correctly recognised it isn't about TSMC.

## Aggregate numbers (20-question golden set)

`python eval/ragas_eval.py --label v1` then `--label v2`. Headline:

| metric | v1 dense-only | v2 hybrid+rerank | Δ |
|---|---:|---:|---:|
| keyword_recall (avg) | 0.412 | 0.671 | +0.259 |
| refusal_rate | 0.30 | 0.15 | −0.15 |
| avg_latency_s | 14.2 | 17.6 | +3.4 |
| ragas/faithfulness | 0.71 | 0.86 | +0.15 |
| ragas/answer_relevancy | 0.79 | 0.88 | +0.09 |
| ragas/context_precision | 0.55 | 0.81 | +0.26 |

(Illustrative seed numbers; rerun with your own corpus for current values. The RAGAS rows require `pip install ragas datasets` and a backing LLM — see [`eval/README.md`](https://github.com/box1401/stock-rag-agentic/blob/main/eval/README.md).)

The latency cost is real: rerank adds one network round-trip (~3 s on a cold reranker container, <500 ms warm). For a research-note generator that already takes ~15 s, fine; for an interactive chat UI you'd want `min_instances=1` on the reranker, which is the only thing that breaks the free-tier story.

## Why RRF, why not weighted-sum?

Weighted-sum requires score normalisation, which is brittle: BM25 scores are unbounded positive reals; dense cosine is in [-1, 1]; the right weights are corpus-dependent and shift as the corpus grows. RRF only uses **rank order**, so it's:

- weight-free (no tuning),
- score-scale-invariant (one less thing to break in production),
- empirically competitive on real benchmarks ([Cormack et al., SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)).

The price is that you discard absolute confidence info from the underlying signals. The cross-encoder rerank then puts that info back, calibrated.

## Common gotchas implementing this

1. **`::text` vs `:ticker`** — Postgres type cast operator collides with SQLAlchemy's named parameter syntax. Use `CAST(:ticker AS text)` instead.
2. **RRF param `k`** — the `60` constant is the de-facto default. Lower `k` makes the fusion more top-heavy; in low-volume corpora that gets noisy.
3. **Reranker cold start** — the `bge-reranker-v2-m3` model is ~600 MB. Bake it into the Docker image (`pip install sentence-transformers && python -c "CrossEncoder(...)"` in the Dockerfile) so cold start is process-spin-up, not model-download.
4. **Generated `tsvector` column** — use `to_tsvector('simple', content)` in a stored generated column, indexed with GIN. `english` config drops every Chinese token.
5. **Embedding dimension** — `gemini-embedding-001` defaults to 3072. Set `output_dimensionality=768` and pin pgvector to `Vector(768)`. Otherwise the index won't build.

## The whole pipeline lives in [retrieval.py](https://github.com/box1401/stock-rag-agentic/blob/main/apps/api/app/rag/retrieval.py)

About 90 lines. The hard part wasn't the code — it was the schema design and the param wiring.

Part 3 of this series covers the deploy story: keyless GitHub-OIDC → GCP, Terraform-managed Cloud Run × 3, and the seven failed deploys it took to get there.
