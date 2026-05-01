---
title: Projects
icon: fas fa-folder-open
order: 3
---

A curated list of what I've shipped. All public on [GitHub](https://github.com/box1401).

---

## torchgeo PR #3655 — Lee filter as a Kornia transform
**Open-source contribution** · [PR #3655](https://github.com/torchgeo/torchgeo/pull/3655) · `Python · PyTorch · Kornia · pytest`

Proposed and implemented the **Lee speckle filter** as `torchgeo.transforms.LeeFilter`, wrapped as a Kornia `IntensityAugmentationBase2D` so it composes naturally with other augmentations in PyTorch pipelines. Currently in review with five `torchgeo` maintainers; 28 tests passing, 100% coverage on the new module.

> Started as Issue [#3645](https://github.com/torchgeo/torchgeo/issues/3645) (proposal) → maintainer green-lit → PR sent.

---

## stock-rag — Taiwan-stock weekly fundamental report generator
**Production RAG** · [box1401/stock-rag](https://github.com/box1401/stock-rag) · `Python · Ollama (Qwen2.5) · TWSE API · Jina · LINE`

End-to-end RAG pipeline: scrapes Taiwan Stock Exchange API + news (DDG + Jina reader) for 21 listed stocks, computes technical indicators, and feeds context into a **local** Ollama LLM to generate per-ticker weekly fundamental reports. Multi-threaded crawler, signal-history CSV for backtesting, optional LINE push notifications. 168 files, ~1400 lines in the main pipeline, 157 sample reports under `Daily_Reports/`.

> Why local LLM? Cost control, privacy, and full reproducibility of the RAG context.

---

## ninjapause — Gesture-controlled Spotify playback
**Computer vision** · [box1401/ninjapause](https://github.com/box1401/ninjapause) · `MediaPipe · OpenCV · Spotify API · PyAutoGUI`

Real-time hand-gesture recognition controlling Spotify playback via webcam. Uses **MediaPipe Hands** for 21-landmark detection; custom finger-extension counter and pinch-distance heuristic with debounce/cooldown to suppress false triggers. Dual control mode: precise Spotify Web API (volume %) + system media keys (works with any player).

---

## refrigerater — LINE-bot smart fridge with Gemini vision
**Application engineering** · [box1401/refrigerater](https://github.com/box1401/refrigerater) · `Flask · Google Gemini · LINE Bot · Sheets API · Docker`

Photograph food → automatic recognition + categorization (ingredients / seasoning / drinks / other) → logged to Google Sheets with expiry tracking. Daily expiry checks, auto-deletion of expired items, push notifications. Containerized with Docker.

---

## ebook-reader — EPUB-to-audiobook converter
**Speech / web app** · [box1401/ebook-reader](https://github.com/box1401/ebook-reader) · `Streamlit · Edge TTS · Google Cloud TTS · ChatTTS + PyTorch`

Streamlit web app that parses EPUB books (TOC, chapter detection, HTML cleanup) and synthesizes audiobooks via three interchangeable TTS engines: Edge TTS (free), Google Cloud WaveNet (high quality), ChatTTS on local GPU. Smart text chunking + audio segment concatenation.

---

## Master's Thesis (in progress)
**Research** · NCU CSRSR · *Comparative Evaluation of Speckle Noise Reduction Techniques for Deep-Learning-Based Ship Detection in SAR Imagery*

Systematic comparison of Lee, Refined Lee, and Frost filters as preprocessing for deep-learning ship detectors on SAR imagery. Advisor: Dr. Hsuan Ren. Defending March 2026.
