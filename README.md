# SalesWhy

**Why did your sales drop? SalesWhy tells you — in plain words, backed by numbers.**

Upload an orders CSV from Shopify, Etsy, or any small shop export. SalesWhy predicts what sales *should* have been, finds the days where they fell short (or jumped), names the most likely cause, and explains it in a short, plain-language note — in English or Telugu.

Built for the [DEV Hacktoberfest Open-Source AI Challenge](https://dev.to) — Week 2 ("Side Hustle").

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![TabPFN](https://img.shields.io/badge/forecasting-TabPFN-7B2FF7.svg)](https://github.com/PriorLabs/TabPFN)
[![Gemma via Ollama](https://img.shields.io/badge/explanations-Gemma%20(local)-4285F4.svg)](https://ollama.com/library/gemma3)
[![Hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2026-ff6600.svg)](https://hacktoberfest.com)

---

## What it does

1. **Upload** an orders CSV (or click "Try sample shop" to see it work instantly).
2. SalesWhy **auto-maps your columns** (date, amount, quantity, product) and lets you confirm or fix the guess.
3. It predicts expected daily revenue with **TabPFN**, trained on your own earlier data.
4. It flags windows where actual revenue was well above or below expected, and **deterministically** ranks price, shipping cost, and product availability as the likely cause — no LLM guessing involved.
5. A local **Gemma** model (via Ollama) turns the finding into three plain sentences a shop owner can actually use.
6. You can ask **"what if?"** (if the price hadn't changed, what would sales have looked like?), **drill into which products** drove the swing, and **save a shareable ledger slip** as a PNG.

## Why it's built this way

- **Deterministic detection, LLM only phrases.** The cause (price, freight, stock) is decided by comparing plain numbers before/after a window — never by asking a language model to guess. Gemma only writes the sentences describing a decision that's already been made.
- **Local-first, and honest about what isn't.** TabPFN and Gemma both run locally on this machine (via the `tabpfn` package and Ollama) — no data leaves it during prediction or explanation. The footer badge in the app always states exactly what ran where, read from the live config, not hardcoded.
- **Every claim has a number behind it.** See [`results/eval.md`](results/eval.md) for the full accuracy table, baselines, and honest discussion of where this approach wins and loses.

## Screenshots

> _Add a screenshot or GIF of the app here before sharing the repo — e.g. `static/screenshot.png`._

## Evaluation

Full table and discussion in [`results/eval.md`](results/eval.md). Headline numbers, from 30 synthetic drops injected into real [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) seller data with known ground-truth causes:

| Pipeline | Overall accuracy | Notes |
|---|---|---|
| SalesWhy (TabPFN + deterministic attribution) | 36.7% | Strong on stockout (90%), weak on price/freight on sparse data |
| Baseline (a): moving-average predictor | 30.0% | Same cause-attribution logic, simpler forecaster |
| Baseline (b): generic LLM, one plain question | **53.3%** | Surprising — see below |

**The most surprising result:** a generic LLM given the same raw data and a single unstructured question *outperformed* SalesWhy's own deterministic pipeline. It reasoned well about stock-outs in natural language, and did moderately better on price/freight too — but it also sometimes gave a confidently wrong answer (blaming "orders dropped" instead of the actual price change it caused), a failure mode SalesWhy's deterministic attribution can't produce by construction. Full discussion, including a grading bug caught mid-evaluation, is in `results/eval.md` and `DEVLOG.md`.

## Tech stack

| Layer | Tool |
|---|---|
| Forecasting | [TabPFN](https://github.com/PriorLabs/TabPFN) (local), gradient-boosting fallback |
| Explanations | [Gemma 3](https://ollama.com/library/gemma3) via [Ollama](https://ollama.com) (local) |
| Backend | FastAPI |
| Frontend | Vanilla JS + [Chart.js](https://www.chartjs.org/), no framework |
| Ledger slip export | Pillow (server-rendered PNG) |
| Evaluation baseline | Google Gemini API |

## Getting started

```bash
git clone https://github.com/Yasaswini-ch/SalesWhy.git
cd SalesWhy
pip install -r requirements.txt

# Pull the local Gemma model (requires Ollama: https://ollama.com)
ollama pull gemma3:4b

# Optional: for baseline (b) evaluation only
cp .env.example .env   # add a GEMINI_API_KEY or OPENAI_API_KEY if you want to re-run it

python app.py
# -> open http://127.0.0.1:8000
```

Click **"Try sample shop"** to see a full analysis instantly, no upload needed.

### Re-running the evaluation pipeline

```bash
python src/prep.py          # aggregate the Olist dataset (needs data/olist_*.csv, see below)
python src/features.py
python src/predict.py
python src/inject.py        # synthetic drops with known ground truth
python src/predict_injected.py
python src/detect.py
python src/eval.py          # writes results/eval.md and results/eval.csv
```

The Olist CSVs aren't included in this repo (`data/` is gitignored) — download them from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) if you want to reproduce the evaluation from scratch.

## Project structure

```
SalesWhy/
  app.py                # FastAPI backend
  static/index.html      # single-page frontend
  src/
    mapper.py             # CSV column auto-mapper
    prep.py, features.py  # Olist data prep + feature engineering
    predict.py            # TabPFN forecasting (local, with fallback)
    detect.py             # deterministic shortfall/spike detection + cause attribution
    explain.py             # Gemma explanations (local, via Ollama)
    inject.py, eval.py     # synthetic evaluation harness
    baseline_llm.py        # baseline (b): generic LLM comparison
    whatif.py               # counterfactual "what if" estimates
    drilldown.py            # top-products-by-impact breakdown
    slip.py                  # shareable ledger slip (PNG) renderer
  results/                # evaluation outputs (eval.md is the one to read)
  tests/fixtures/         # sample CSVs used to test the column mapper
  DEVLOG.md               # full build log, including what broke and why
```

## Known limitations

- Cause attribution (price/freight specifically, not stock-outs) is noticeably weaker on sparse, low-volume seller data — documented honestly in `results/eval.md` rather than hidden.
- Gemma explanations take 15-20s per call on CPU (no GPU in this environment); the app only generates one live explanation per analysis and fetches others on demand.
- `n_orders` in the generic CSV pipeline counts line items, not deduplicated orders, for CSVs without a distinct order-ID column.

See `DEVLOG.md` for the full build history, including every bug found and how it was fixed.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC-BY-NC-SA) for development and evaluation data.
- [UCI/Kaggle Online Retail dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail) for real-world (non-synthetic) pipeline testing.
- [TabPFN](https://github.com/PriorLabs/TabPFN), [Gemma](https://ai.google.dev/gemma), [Ollama](https://ollama.com), [Chart.js](https://www.chartjs.org/).
