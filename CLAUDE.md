# CLAUDE.md: SalesWhy

## Project
**SalesWhy** explains, in plain words, why a tiny online shop's sales dropped. The user uploads an orders CSV. The tool predicts what sales *should* have been, finds windows where they fell short, names the likely cause, and writes a short explanation for a shop owner.

Built for the DEV Hacktoberfest Open-Source AI Challenge: Week 2 (theme: Side Hustle).
- Deadline: Oct 18, 2026, 11:59 PM PDT (about 12:30 PM IST on Oct 19). Target publish: evening of Oct 18 IST.
- Judging: writing quality (heaviest), relevance, creativity, technical execution, partner tech.
- Partner categories targeted: Best Use of TabPFN, Best Use of Gemma. Optional: .tech domain, Render.
- Required post tags: #devchallenge #hf26challenge

## Principles
1. **Deterministic detection, LLM only phrases.** Cause attribution is plain code. Gemma turns structured findings into sentences. Never let the LLM decide the cause.
2. **Measure everything.** Every claim in the write-up needs a number from `results/`.
3. **Be honest about what runs where.** TabPFN-3.5 uses an API (data leaves the machine). Document exactly which parts are local and which are not. Never claim "100% local" unless true.
4. **Keep notes as you go** in `DEVLOG.md`. Failures are valuable material for the post.
5. **Colab T4 friendly.** Save intermediate outputs to Drive after every step. Sessions disconnect.

## Data
- Base: Olist Brazilian E-Commerce dataset (Kaggle). Join orders, order_items, products.
- Pick the top 5 sellers by order count. Aggregate to daily revenue per seller.
- Synthetic drops with known causes (ground truth) for evaluation: 15% price hike, 5-day stockout, freight spike, promo removal.
- Real user data: one real seller's orders export (Shopify or Etsy CSV). Never commit it. Delete after the demo.
- `data/` is gitignored except a small sample shop CSV used by the "Try sample shop" button.

## Pipeline
1. `prep.py`: joins and daily aggregation per seller.
2. `features.py`: day of week, month, avg price, avg freight, distinct active products, days since last price change.
3. `predict.py`: TabPFN predicts expected daily revenue. Train on earlier dates, predict later ones. Fallback: local TabPFN package, then gradient boosting (state which was used).
4. `detect.py`: flag windows where actual revenue is well below predicted for 2+ consecutive days. For each candidate cause (price up, freight up, fewer active products, calendar effect), compare values inside vs outside the window. Rank by size of change. Output structured JSON like:
   `{"window": "2026-10-03..05", "shortfall_pct": 32, "cause": "price_increase", "change_pct": 14, "confidence": 0.8}`
5. `inject.py`: add known drops to held-out periods and record ground truth.
6. `explain.py`: Gemma converts each JSON finding into 3 plain sentences. Support English and Telugu. Use a smaller Gemma variant if the T4 is too slow.
7. `app.py`: FastAPI backend plus a single static `index.html`.

## Features (build in tiers, never start a tier early)

**Rule:** the core pipeline, the evaluation table and the basic UI come first. Do not start a tier until the one above it works and is logged in DEVLOG.md. If time is short, cut from the bottom. The write-up matters more than any feature.

### Tier 1: required (finish by end of Oct 16)
1. **CSV column auto-mapper** (`src/mapper.py`)
   - Detect date, amount, quantity and product columns from Shopify, Etsy and Olist-style exports using header names and value patterns.
   - Show the guess in the UI with one dropdown per column so the user can correct it.
   - Test with at least 3 differently shaped CSVs before moving on.
2. **"Not sure" state** (in `detect.py` and the UI)
   - If no candidate cause explains the drop above a confidence threshold, output `{"cause": "unknown"}`.
   - UI copy: "Sales dipped X%, but nothing in your data explains it. It may be something outside this file, like a competitor or an ad change."
   - Report the abstain rate in `results/eval.md`.
3. **Shareable ledger slip** (`src/slip.py`)
   - One button exports the findings as a PNG or PDF in the paper-and-ink style (same colors and fonts).
   - Must look good when shared on WhatsApp (portrait, readable at phone size).

### Tier 2: only if Tier 1 is done by Oct 16 (pick in this order)
1. **What-if slider** (`src/whatif.py`)
   - Change one feature (price, freight) and re-predict with TabPFN: "If you had kept the price the same, sales would have been about X."
   - Show the result as a dashed alternate line on the chart. Label it clearly as an estimate.
2. **Product drill-down**
   - Clicking a finding shows the top 3 products by lost revenue in that window.
3. **Upside detection**
   - Detect and explain spikes too ("Sales jumped 40% after the weekend promo"). Reuse `detect.py` with the sign flipped. Green accent, never red.

### Tier 3: only if everything above is done, capped at 1 hour each
1. **Sentry agent tracing:** log latency, tokens and cost per explanation. Put screenshots in the post to support the "open models are cheap" argument.
2. **ElevenLabs voice readout** of the Telugu explanation. Check cost first. Skip if it takes more than an hour.
3. **"What ran where" footer badge:** reads the actual config and shows TabPFN: API or local, Gemma: local or hosted. Never hardcode it.

### Do NOT build
User accounts, multi-chart dashboards, live Shopify or Etsy integrations, notifications, multi-shop support, or a settings screen. None of these helps the evaluation or the write-up.

## Evaluation (must be produced)
Write to `results/eval.md` and `results/eval.csv`:

| Injected cause | Times injected | Correctly named | Accuracy |
|---|---|---|---|
| Price hike | 30 | ? | ? |
| Stockout | 30 | ? | ? |
| Freight spike | 30 | ? | ? |

Also report the abstain rate (how often the tool said "not sure") and how many of those were truly unexplainable drops versus misses.

Also compare against baselines: (a) moving-average predictor, (b) a generic LLM asked "why did sales drop?" with the same data. Report where SalesWhy wins and where it does not.

## Repo structure
```
saleswhy/
  CLAUDE.md
  DEVLOG.md
  README.md
  data/
  src/
    prep.py
    features.py
    mapper.py      # CSV column auto-mapping
    predict.py
    detect.py
    inject.py
    explain.py
    slip.py        # ledger slip export
    whatif.py      # Tier 2: counterfactual predictions
  app.py
  static/
    index.html
  notebooks/
  results/
```

## Design system: "shopkeeper's ledger"
Warm paper, ink text, one stamp-red accent. It should feel like a clean modern ledger page, not a default AI app.

**Colors (use only these):**
| Role | Hex |
|---|---|
| Paper (background) | `#F6F1E7` |
| Card surface | `#FBF8F1` |
| Ink (text) | `#1F1B16` |
| Muted text | `#6B6358` |
| Hairline borders | `#D9D0BE` |
| Drop accent (red) | `#B3382C` |
| Positive (green) | `#3E6B4A` |
| Highlight (marigold) | `#E0A526` |

Red is reserved for drops and problems only.

**Fonts (Google Fonts):**
- Headings: Fraunces
- Body: Instrument Sans
- Numbers and dates: IBM Plex Mono
- Regional text fallback: Noto Sans Telugu (and Noto Sans Devanagari if Hindi is added)

**Style rules:**
- Radii 4 to 6px. 1px borders instead of shadows. Generous whitespace.
- NO gradients, glassmorphism, glowing buttons, purple or blue AI palette, sparkle icons, emoji icons, or centered-hero-plus-three-cards layout. No Inter on white. No Streamlit or Gradio default look.
- Mobile-first. Test at 390px width. Body text 16px or larger. High contrast.

**Layout (one page, one flow):**
1. Header: "SalesWhy" in Fraunces plus one line of explanation.
2. Upload: large dashed-border box, "Drop your orders CSV here", plus a "Try sample shop" button so judges can test instantly. After upload, show the column auto-mapper guess with one dropdown per column.
3. Chart (Chart.js or uPlot, custom styled): actual sales as a solid ink line, expected as a dashed muted line, drop windows shaded light red, key dips labeled directly on the chart (no legend).
4. Findings as ledger rows, for example `03-05 Oct · Sales 32% below normal · Likely cause: price rose 14%`. Dates and numbers in mono. Clicking a row highlights that window on the chart. Unknown causes use the "not sure" copy.
5. Explanation card styled like a handwritten note, 3 sentences max, with an English/Telugu toggle.
6. "Save ledger slip" button.
7. Footer: one small honest line about what runs where.

**UX rules:** no settings screen (inputs are the upload, the column dropdowns and the language toggle only). Clear loading state ("Reading your orders... finding the dips..."). Plain-language error if CSV columns cannot be mapped. Must work without sign-in.

## DEVLOG.md (maintain throughout)
After each work session, append:
- What I built or tried
- What failed or surprised me (with specifics)
- Numbers (accuracy, latency, cost)
- Decisions and why

Also log the first-user interaction: who, what they said, whether they would pay, what they asked for next.

## Timeline
| Day | Goal |
|---|---|
| Oct 13 | Load Olist, choose sellers, daily aggregation, send export requests to 3 to 5 real sellers |
| Oct 14 | Features plus TabPFN model, start `mapper.py` |
| Oct 15 | Synthetic drops, evaluation, baselines, "not sure" state |
| Oct 16 | Gemma explanations, web UI, ledger slip. **Cut line: Tier 1 must be done.** |
| Oct 17 | Run on a real seller's export, one Tier 2 feature (what-if), start the post |
| Oct 18 | No new features. Finish the post, deploy, publish by evening IST |

## Fallbacks
- TabPFN API trouble: local TabPFN package, then gradient boosting. Say which in the post.
- Gemma too slow: smaller variant, or explain only the top finding.
- No real seller replies: use a public small-shop dataset and say so plainly.
- Colab disconnects: checkpoint to Drive after every step.

## Blog post outline (writing quality is weighted most)
1. Hook: a specific scene of a seller staring at a dip.
2. Who would pay and why (one narrow buyer).
3. Why open mattered (cost per query, privacy, swappable models), including where it did not.
4. How it works (one diagram, short).
5. Results table, abstain rate and baseline comparison.
6. What failed (pull from DEVLOG.md).
7. First user: reaction, willingness to pay, next request.
8. What's next.

Optional extras: embed the Claude Code session via DevRelay; link the live demo; host on a free .tech domain from hacktoberfest.com/my/promos.

## Working rules for Claude Code
- Start by reading this file, then work on the current day's tasks only (see Timeline). Do not jump ahead to later tiers.
- Build in the order of the pipeline. Verify each stage with a small test before moving on.
- Keep code simple and readable. No unnecessary frameworks.
- Never commit API keys or real user data. Use `.env` and `.gitignore`.
- After each milestone, update DEVLOG.md and results/.
- When a decision is ambiguous, pick the simplest option, log it in DEVLOG.md, and continue.
