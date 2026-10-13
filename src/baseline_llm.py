"""Baseline (b) from CLAUDE.md's evaluation section: a generic LLM asked
"why did sales drop?" with the same data, no structured prompting, no
deterministic cause attribution, no engineering beyond a single plain
question -- the point is to measure what SalesWhy's pipeline buys over just
pasting data into a chat model.

Classifies the model's free-text answer into one of the known causes with
simple keyword matching (the same thing a human reading the answer would
do), so it can be scored against inject.py's ground truth the same way the
TabPFN and moving-average baselines are.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-3.1-flash-lite"
CONTEXT_DAYS_BEFORE = 14

KEYWORDS = {
    "price_increase": ["price", "pricing", "cost increase", "expensive", "raised"],
    "freight_spike": ["shipping", "freight", "delivery cost", "postage"],
    "stockout": [
        "stock", "inventory", "out of stock", "unavailable", "sold out",
        "active products", "no products", "zero products", "lack of products",
        "without any products", "products listed", "products available",
        "products for sale",
    ],
}


def _classify(answer: str) -> str:
    text = answer.lower()
    if any(p in text for p in ["not sure", "insufficient", "cannot determine", "can't determine", "unclear", "no clear"]):
        return "unknown"
    scores = {cause: sum(k in text for k in kws) for cause, kws in KEYWORDS.items()}
    top_cause = max(scores, key=scores.get)
    return top_cause if scores[top_cause] > 0 else "unknown"


def _build_context(df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> str:
    start_idx = df.index[df["date"] >= window_start - pd.Timedelta(days=CONTEXT_DAYS_BEFORE)][0]
    end_idx = df.index[df["date"] <= window_end][-1]
    context = df.loc[start_idx:end_idx, ["date", "revenue", "avg_price", "freight", "n_active_products", "n_orders"]]
    lines = ["date,revenue,avg_price,freight,active_products,orders"]
    for row in context.itertuples():
        lines.append(f"{row.date.date()},{row.revenue:.2f},{row.avg_price:.2f},{row.freight:.2f},{row.n_active_products},{row.n_orders}")
    return "\n".join(lines)


def ask_llm(client, csv_blob: str, window_start: str, window_end: str) -> dict:
    prompt = (
        f"Here is my online shop's daily sales data as CSV:\n\n{csv_blob}\n\n"
        f"My sales dropped between {window_start} and {window_end}. Why did sales drop? "
        "Give me the single most likely cause in one short sentence."
    )
    start = time.time()
    last_exc = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            break
        except Exception as exc:  # transient 503s are common on this model
            last_exc = exc
            time.sleep(3 * (attempt + 1))
    else:
        raise last_exc
    latency_s = round(time.time() - start, 2)
    usage = resp.usage_metadata
    return {
        "answer": resp.text.strip(),
        "latency_s": latency_s,
        "prompt_tokens": usage.prompt_token_count,
        "completion_tokens": usage.candidates_token_count,
    }


def main() -> None:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY set -- skipping baseline (b).")
        return
    client = genai.Client(api_key=api_key)

    ground_truth = pd.read_csv(ROOT / "results" / "injected" / "ground_truth.csv", parse_dates=["window_start", "window_end"])
    rows = []
    for gt in ground_truth.itertuples():
        df = pd.read_csv(ROOT / "results" / "injected" / f"{gt.seller_id}.csv", parse_dates=["date"])
        csv_blob = _build_context(df, gt.window_start, gt.window_end)
        try:
            result = ask_llm(client, csv_blob, str(gt.window_start.date()), str(gt.window_end.date()))
        except Exception as exc:
            print(f"  FAILED on {gt.seller_id} {gt.window_start.date()}: {exc}")
            continue
        detected = _classify(result["answer"])
        correct = detected == gt.cause
        rows.append({
            "seller_id": gt.seller_id,
            "window_start": gt.window_start.date(),
            "window_end": gt.window_end.date(),
            "true_cause": gt.cause,
            "llm_answer": result["answer"],
            "detected_cause": detected,
            "correct": correct,
            "abstained": detected == "unknown",
            "latency_s": result["latency_s"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        })
        print(f"{gt.seller_id} {gt.window_start.date()}: true={gt.cause} detected={detected} correct={correct}")

    out = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out.to_csv(out_dir / "baseline_llm.csv", index=False)

    per_cause = out.groupby("true_cause").agg(
        times_injected=("true_cause", "count"),
        correctly_named=("correct", "sum"),
    )
    per_cause["accuracy"] = (per_cause["correctly_named"] / per_cause["times_injected"]).round(3)
    summary = {
        "abstain_rate": round(out["abstained"].mean(), 3),
        "overall_accuracy": round(out["correct"].mean(), 3),
        "mean_latency_s": round(out["latency_s"].mean(), 2),
        "total_prompt_tokens": int(out["prompt_tokens"].sum()),
        "total_completion_tokens": int(out["completion_tokens"].sum()),
    }
    print("\n", per_cause, "\n", summary)

    per_cause.reset_index().to_csv(out_dir / "baseline_llm_summary.csv", index=False)
    import json
    (out_dir / "baseline_llm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
