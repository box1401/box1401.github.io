---
layout: post
title: "Building a Local-LLM RAG Pipeline for Taiwan Stocks"
date: 2026-05-04 19:00:00 +0800
categories: [LLM, Data]
tags: [rag, ollama, qwen2.5, twse, finmind, threading, jina, taiwan-stocks]
---

This post is the architecture walkthrough for [stock-rag](https://github.com/box1401/stock-rag), a side project I built earlier this year that produces weekly investment-style reports for ~21 Taiwan-listed stocks. Everything runs on a single laptop: news crawling, multi-source data fusion, technical indicators, and the LLM that synthesises it all into a Markdown report. There's no cloud LLM, no SaaS pipeline, no monthly bill.

I want to talk about three design choices that made it work end-to-end on my own machine: (1) why I chose **local Ollama + Qwen2.5** over OpenAI, (2) how I fused **7 TWSE endpoints + FinMind + threaded news crawling** into a single context block, and (3) the **threading + rate-limit discipline** that kept TWSE from blocking my IP.

## 1. Why local LLM, not OpenAI

This was the first decision and the one most people second-guess. The argument for OpenAI is straightforward — you get GPT-4-class reasoning at the cost of a few cents per call, no infrastructure to maintain. For a project that calls the model 21 times a week (one per stock, weekly), I'd be paying maybe USD $5/month. Cheap.

I went local anyway, for three reasons that compound:

- **Cost is not the bottleneck — iteration speed is.** When I'm tuning the prompt and want to re-run the full 21-stock pipeline 6 times in an evening to compare prompt variants, hitting a paid API gets slow (rate limits, network round-trips) and the bill creeps. With a local 14B model on my GPU, I can re-run all 21 in ~12 minutes and the cost is electricity.
- **The data going into the prompt is, basically, ALL of TWSE.** Per-stock context is ~30-50KB of structured market data plus 5+ news articles' full text. Sending that to a third-party API for an *experimental* personal project felt like the wrong default — not because TWSE data is private (it isn't), but because the **habit** of routing private-domain analysis through someone else's logs is one I don't want to build for my next project where the data *is* sensitive.
- **The reports are for me.** No SLA. If `ollama serve` falls over because Windows decided to update, I notice when I check the `Daily_Reports/` folder Sunday morning, and I rerun. A SaaS dependency would mean writing retry/fallback logic for an audience of one. Not worth it.

The model I settled on is **Qwen 2.5:14b** via Ollama. The prompt structure is at the bottom of [`auto_report.py`](https://github.com/box1401/stock-rag/blob/main/auto_report.py); the call is a vanilla `POST http://localhost:11434/api/chat`. 14B is the sweet spot — 7B was too inconsistent on the technical-indicator interpretation, 32B was 3× slower without an obvious quality jump for this task. If you have more VRAM, swap the model name in the config and go.

## 2. Multi-source data fusion (7 TWSE endpoints + FinMind + news)

The hard part of a RAG pipeline isn't the LLM — it's getting the right context block ready. For each stock, the system pulls from:

**TWSE official APIs (intraday + market-wide):**
- `T86` — institutional buy/sell trades
- `MI_MARGN` — margin trading volumes
- `MI_INDEX` — market-wide index data
- `BWIBBU_d` — full-market PBR / dividend yield / PE ratio
- `STOCK_DAY` — single-stock monthly daily-K (with fallback to FinMind for OTC stocks)

**FinMind (longer history + statements):**
- `TaiwanStockMonthRevenue` — 14 months of monthly revenue
- `TaiwanStockFinancialStatements` — 8 quarters of income statements (EPS, gross margin, op margin)
- `TaiwanStockPER` — 12 months of daily PE/PBR for valuation banding
- `TaiwanStockDividendResult` — 5 years of dividend history

**News (threaded crawl):**
- `DDGS().news()` for the headline list
- [`r.jina.ai`](https://r.jina.ai) for full-article extraction (the underrated piece — Jina's reader API skips JS-rendered junk and returns clean markdown)

Once everything is fetched, the per-stock context is ~30-50KB of structured Markdown. The LLM gets that in a single prompt with strict instructions:

```python
prompt = f"""
你現在是一個專業的台股研究員，請完全根據以下提供的資料內容，
來分析 {company_name} 的基本面。
如果資料中沒有提到的資訊，請回答「資料中未提及」，
絕對不要自行編造。

【以下是爬蟲抓取的最新資料】：
{content_text}
"""
```

The "**don't make things up — say 'not in the data'**" instruction matters more than people realise. Without it, even the best local models will happily hallucinate quarterly EPS figures with 3 decimal places of false precision.

## 3. Threading + rate-limit discipline

This is the unglamorous part that kept the project working past day 3. TWSE's APIs are public but unforgiving — a few seconds of unthrottled parallel requests and your IP gets a temporary block.

The pattern that worked is **separate locks for separate domains**:

```python
import threading

print_lock = threading.Lock()
twse_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def twse_get(url, timeout=10):
    """All TWSE calls funnel through here for IP-friendly throttling."""
    with twse_lock:
        resp = requests.get(url, headers=TWSE_HEADERS, timeout=timeout)
        time.sleep(0.4)
    return resp
```

The `twse_lock` makes every TWSE request **strictly serial** even when called from multiple worker threads. The 400ms `time.sleep` is the rate budget — too aggressive and TWSE throttles, too conservative and the 21-stock loop takes forever. 400ms came from binary search.

The non-TWSE pieces — news search and Jina article extraction — *do* run in parallel:

```python
NEWS_WORKERS = 3   # DDGS news search
JINA_WORKERS = 4   # r.jina.ai full-article extraction
```

This is a "2-tier" worker model: 3 news searchers feed a queue that 4 article extractors drain. The asymmetry comes from real measurements — Jina extraction is ~2× slower than DDGS search, so the extractor pool needs to be wider to keep the search pool from blocking on a full queue. `print_lock` exists so the per-thread progress logs don't interleave into garbage in the terminal.

The whole 21-stock pipeline runs in ~12 minutes on a residential connection. The TWSE serial throttle is the dominant cost; everything else is parallel.

## 4. Outputs as inputs (the part I underbuilt)

The pipeline writes per-stock Markdown reports to `Daily_Reports/YYYY-MM-DD_<ticker>_{daily,weekly}.md` and pushes to LINE for read-on-phone consumption. It also exports a 23-column CSV of per-stock signals (MA5/10/20/60 cross states, RSI14, BIAS-20, valuation percentile, etc.) for the same date — a thin bridge to backtesting.

The honest gap: **I never closed the loop**. The signals CSV grows weekly but no model consumes it. The next iteration of this project would be to use the signal history to backtest the LLM's *qualitative* calls — does "強勢突破" actually correlate with positive forward returns? Did the LLM reliably miss reversals? That's the natural Refined Lee → ship-detection equivalent for a stock RAG: stop generating reports and start measuring the reports.

## What I'd do differently if I rebuilt today

1. **Move the 30-50KB per-stock context off the prompt and into a vector store**, then retrieve only the 5-10 most relevant chunks per query. Local models start to lose coherence past ~20KB of context; selective retrieval would let me drop to a 7B model with the same quality. (Yes, this is what RAG actually means; the v1 above is "stuff everything into the prompt", which is a degenerate case.)
2. **Persist the model output back to the next prompt as memory** — "last week you said X about this stock, here's what's happened since" — instead of stateless single-shot generation.
3. **Replace the 400ms `time.sleep` throttle with a token-bucket rate-limiter** that adapts when TWSE returns HTTP 429.

The repo as-it-is is a working prototype with a few rough edges (`run_stock_report.bat` has the old hardcoded path, the secret keys are placeholders so you'd need to drop in your own Gemini / Ollama setup). It's [github.com/box1401/stock-rag](https://github.com/box1401/stock-rag) if you want to fork it as a starting point.

If you've shipped a similar local-LLM analysis pipeline I'd love to compare notes — especially on the "how to actually evaluate hallucinations against ground-truth market data" piece, which is the part I haven't solved.
