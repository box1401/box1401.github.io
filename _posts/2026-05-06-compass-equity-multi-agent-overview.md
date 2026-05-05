---
layout: post
title: "Building Compass Equity — a multi-agent equity research copilot for Taiwan stocks"
date: 2026-05-06 19:00:00 +0800
categories: [LLM, Engineering]
tags: [agents, rag, langgraph, fastapi, nextjs, gcp, cloudrun, gemini, taiwan-stocks]
---

> Most "equity research copilots" are single-shot RAG demos. Compass Equity is built around three product principles: multi-agent with adversarial review, hybrid retrieval (not vector-only), and full auditability. This post explains why those choices matter and how the system fits together.

## TL;DR

- One ticker in (e.g. `2330`) → grounded research note out, with citations and a live agent trace.
- Five agents: **Supervisor → Data → Analyst → Risk → Reporter**, plus a self-critique loop where Risk can demand a rewrite.
- The Analyst can dynamically call `rag_search`, `news_search`, and `read_url` via Gemini function calling — it decides what context it needs.
- Live demo on Cloud Run: <https://compass-web-aujzogkiva-de.a.run.app>
- Source: <https://github.com/box1401/stock-rag-agentic>

## Why "agentic", and not just RAG?

A vanilla RAG pipeline glues retrieval to a single LLM call:

```text
question → retrieve top-k → stuff into context → answer
```

This works for fact lookup. It breaks down on equity research, because the answer needs to fuse:

1. **Time-series numbers** — last close, MA20, RSI, foreign net buy/sell — that are not in a vector index.
2. **Static thesis context** — a long-form bull case, the macro overlay, position-sizing rules — which IS in the vector index.
3. **Latest news catalysts** — earnings calls, regulatory filings — which need a fresh web search.
4. **Adversarial review** — does this draft actually support every number it cites, or is the LLM smoothing over gaps with confident prose?

Doing all four well in one prompt produces brittle output. Splitting into specialised agents gives each one a tight contract and a tight prompt.

## The five-agent flow

```text
                  ┌─── Supervisor ───┐
                  │                  │
             dispatches         orchestrates
                  │                  │
                  ▼                  │
              ┌─Data─┐                │
              │ TWSE / FinMind / Yahoo│
              └─────┬─┘              │
                    ▼                │
              ┌─Analyst─┐  ◄──────────┘
              │ function calling:    │
              │   rag_search         │
              │   news_search        │
              │   read_url           │
              └─────┬─┘              │
                    ▼                │
              ┌──Risk──┐             │
              │ critiques the draft  │
              │ pass / issues[]      │
              └─────┬─┘              │
                    │ if !pass and revisions < 2
                    ▼                │
              ┌─Revise─┐             │
              │ analyst rewrites     │
              └─────┬─┘─────────────►│ back to Risk
                    ▼ pass
              ┌─Reporter─┐
              │ markdown + citations │
              └─────┬─┘
                    ▼
                  user
```

Implemented with [LangGraph](https://langchain-ai.github.io/langgraph/). State is a `TypedDict`; nodes are async functions; edges are static or conditional. The conditional edge on `should_revise` is the actual self-critique loop.

## What the Risk agent catches

In testing on the 20-question golden set, RiskAgent flags about 15 % of drafts on first pass. The most common issues:

- **Unsupported precision**: "gross margin will expand to 58.4 %" when the source thesis only says "58–60 %".
- **Single-sided framing**: bull case stated without acknowledging the explicitly listed downside risks.
- **Ambiguous hedging**: "may, perhaps, looks like" used to dress up "we have no data".

A revised draft after one rewrite usually clears the second pass. After two failed passes, the system gives up and ships the original draft with a `## Risk Notes` section appended — better than infinite-looping.

## Why Gemini function calling and not ReAct prompts?

Two reasons.

1. **The schema is hard for the model to break**. Gemini's function-calling API validates arguments; a malformed call returns an error to the agent rather than crashing the runtime.
2. **You don't pay for a parsing dance**. Compared to ReAct, where the model emits "Thought / Action / Action Input" text that you regex-parse, function calling is half the tokens and zero parser code.

The Analyst agent's tool loop is short:

```python
for step in range(MAX_ITER):  # MAX_ITER = 5
    result = await gateway.generate(messages, tools=tools)
    if not result.tool_calls:
        parsed = _extract_json(result.text)
        break
    messages.append(ChatMessage(role="assistant", tool_calls=result.tool_calls))
    for tc in result.tool_calls:
        tool_text, tool_cites = await _exec_tool(tc, ticker)
        messages.append(ChatMessage(role="tool", name=tc.name, content=tool_text))
```

In the wild, a typical run uses 1–2 tool calls and two LLM round-trips total: one to plan, one to write the JSON answer.

## The stack, and why each piece

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 App Router + shadcn/ui | SSR for the landing; client EventSource for the live trace UI |
| Backend | FastAPI + Pydantic v2 + SQLAlchemy 2.0 async + Alembic | async-native, strict typing, mature migration story |
| LLM | Gemini 2.5 Flash (primary) + Flash Lite (fallback) | free tier on AI Studio; fallback handles transient quota |
| Embeddings | `gemini-embedding-001` truncated to 768 dims | matches pgvector schema; same SDK as the LLM |
| Reranker | `bge-reranker-v2-m3` (cross-encoder) | order-of-magnitude better than identity ordering; free, runs on CPU |
| RAG | Postgres `tsvector` BM25 + pgvector dense + RRF fusion + cross-encoder rerank | hybrid beats either signal alone on low-volume corpora |
| Agents | LangGraph 1.x | first-class conditional edges, async nodes, state-as-TypedDict |
| DB | Supabase Postgres + pgvector | free tier 500 MB, matches the local docker-compose pgvector image |
| Auth | Supabase Auth (Google OAuth) | shared with the DB, free, no extra service |
| Eval | RAGAS + a 20-question golden set | catches regressions in faithfulness and context precision |
| Tracing | Langfuse Cloud (free tier) | per-run trace tree of pipeline → agent → LLM call |
| Deploy | Cloud Run × 3 (api / web / reranker) + Terraform + WIF | serverless scale-to-zero; keyless GitHub OIDC deploy |

## Observability — every claim is traceable

Each `/analyze` call carries a `trace` array in the response. The frontend `/trace` page subscribes to a server-sent-event stream and shows the events as they happen:

```text
17:01:22  supervisor  dispatch
17:01:22  data        fetch_start
17:01:26  data        fetch_done
17:01:26  analyst     indicators_computed
17:01:30  analyst     tool:rag_search    args={"query":"investment thesis for TSMC", "ticker":"2330"}  2596 chars
17:01:33  analyst     tool:news_search   args={"query":"TSMC AI accelerator demand"}  580 chars
17:01:38  analyst     draft_done
17:01:42  risk        review_done        pass=true  issues=0
17:01:42  reporter    report_rendered
```

When Langfuse is wired in, the same tree shows up in the Langfuse UI, with the LLM input/output text on each generation node and the latency/usage numbers attached. Regression hunting becomes tractable.

## What's next

- **M4** — RAGAS-backed eval gate in CI; A/B graphs for v1 vs v2.
- Pre-baked PDF ingestion for MOPS filings (Taiwan public-disclosure system).
- Web demo with a curated "watchlist" page.

If you want to skim the code, the entry points are:

- [`apps/api/app/agents/supervisor.py`](https://github.com/box1401/stock-rag-agentic/blob/main/apps/api/app/agents/supervisor.py) — LangGraph wiring
- [`apps/api/app/agents/analyst_agent.py`](https://github.com/box1401/stock-rag-agentic/blob/main/apps/api/app/agents/analyst_agent.py) — function-calling tool loop
- [`apps/api/app/rag/retrieval.py`](https://github.com/box1401/stock-rag-agentic/blob/main/apps/api/app/rag/retrieval.py) — hybrid search + RRF + rerank
- [`infra/terraform`](https://github.com/box1401/stock-rag-agentic/tree/main/infra/terraform) — Cloud Run + WIF

Part 2 of this series digs into the RAG numbers — A/B between dense-only, BM25-only, and hybrid-with-rerank — using the RAGAS golden set as the scoreboard.
