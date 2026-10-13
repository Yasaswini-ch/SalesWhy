"""Turn a detect.py finding into a short, plain-language explanation.

Gemma (local, via Ollama) only phrases a finding that detect.py already
decided deterministically -- it never sees raw data and never chooses the
cause (CLAUDE.md principle 1). The "not sure" case bypasses the LLM
entirely and uses fixed copy, since consistency matters more than
variation for a message this important.
"""
from __future__ import annotations

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

CAUSE_LABELS = {
    "price_increase": "the price went up",
    "freight_spike": "shipping cost went up",
    "stockout": "products were unavailable to sell",
    "price_decrease": "the price came down",
    "freight_decrease": "shipping cost came down",
    "more_products": "more products were available to sell",
}

UNKNOWN_COPY = {
    "dip": {
        "en": "Sales dipped {shortfall_pct:.0f}%, but nothing in your data explains it. It may be something outside this file, like a competitor or an ad change.",
        "te": "అమ్మకాలు {shortfall_pct:.0f}% పడిపోయాయి, కానీ మీ డేటాలో దానికి కారణం కనిపించలేదు. ఇది మీ ఫైల్ బయట ఉన్న విషయం కావచ్చు, ఉదాహరణకు పోటీదారు లేదా ప్రకటనల్లో మార్పు.",
    },
    "spike": {
        "en": "Sales jumped {shortfall_pct:.0f}%, but nothing in your data explains it. It may be something outside this file, like a promo or some press.",
        "te": "అమ్మకాలు {shortfall_pct:.0f}% పెరిగాయి, కానీ మీ డేటాలో దానికి కారణం కనిపించలేదు. ఇది మీ ఫైల్ బయట ఉన్న విషయం కావచ్చు.",
    },
}


def _call_ollama(prompt: str, timeout: int = 60) -> str:
    # num_predict caps generation length -- 3 short sentences never need more
    # than ~120 tokens, and capping it keeps a slow CPU decode from running on.
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 120},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["response"].strip()


def _build_prompt(finding: dict, language: str) -> str:
    cause_label = CAUSE_LABELS.get(finding["cause"], finding["cause"])
    lang_instruction = "Write in Telugu." if language == "te" else "Write in plain English."
    is_spike = finding.get("direction") == "spike"
    direction_phrase = "were" + (f" {finding['shortfall_pct']:.0f}% above" if is_spike else f" {finding['shortfall_pct']:.0f}% below") + " what was expected"

    return (
        "You are writing for a small shop owner who is not technical. "
        f"Between {finding['window_start']} and {finding['window_end']}, their sales {direction_phrase}. "
        f"The data shows this happened because {cause_label} (it moved by {finding['change_pct']:.0f}% in that window). "
        f"{lang_instruction} Write exactly 3 short sentences, friendly and plain, explaining what "
        "happened and " + ("noting this is good news they could try to repeat" if is_spike else "gently suggesting one thing they could look at") + ". "
        "Do not mention percentages with more than one decimal. Do not invent any cause other than the one given. "
        "Do not add a greeting or sign-off -- just the 3 sentences."
    )


def explain(finding: dict, language: str = "en") -> dict:
    direction = finding.get("direction", "dip")
    if finding["cause"] == "unknown":
        text = UNKNOWN_COPY[direction][language].format(shortfall_pct=finding["shortfall_pct"])
        return {"text": text, "backend": "fixed-copy", "latency_s": 0.0}

    prompt = _build_prompt(finding, language)
    start = time.time()
    try:
        text = _call_ollama(prompt)
        backend = f"ollama:{MODEL}"
    except Exception as exc:  # pragma: no cover - environment dependent
        verb = "above" if direction == "spike" else "below"
        text = (
            f"Sales were {finding['shortfall_pct']:.0f}% {verb} expected because {CAUSE_LABELS.get(finding['cause'], finding['cause'])}."
        )
        backend = f"fallback-template ({exc!r})"
    latency_s = round(time.time() - start, 2)
    return {"text": text, "backend": backend, "latency_s": latency_s}


if __name__ == "__main__":
    sample = {
        "seller_id": "demo",
        "window_start": "2018-05-15",
        "window_end": "2018-05-16",
        "shortfall_pct": 51.9,
        "cause": "stockout",
        "change_pct": 43.5,
        "confidence": 0.74,
    }
    for lang in ("en", "te"):
        result = explain(sample, language=lang)
        print(lang, "->", result)
