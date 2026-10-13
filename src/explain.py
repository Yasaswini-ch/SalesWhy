"""Turn a detect.py finding into a short, plain-language explanation.

Gemma only phrases a finding that detect.py already decided deterministically
-- it never sees raw data and never chooses the cause (CLAUDE.md principle 1).
The "not sure" case bypasses the LLM entirely and uses fixed copy, since
consistency matters more than variation for a message this important.

Backend: local Ollama first (the real "100% local" path, used when this app
runs on your own machine). If Ollama isn't reachable -- e.g. this is
deployed to Render, which doesn't run a local Ollama daemon alongside the
web service -- it falls back to the *same* Gemma model family hosted via
the Gemini API, and only after that to a fixed template. Which one actually
ran is always reported back in `backend`, never hidden.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_PING_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_PING_TIMEOUT = 1.5  # fail fast if there's no local Ollama at all (e.g. on Render)
OLLAMA_GENERATE_TIMEOUT = 60  # once we know Ollama is up, a cold model load can be slow
MODEL = "gemma3:4b"
HOSTED_GEMMA_MODEL = "gemma-4-26b-a4b-it"

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


def _ollama_available() -> bool:
    # A cold Gemma load can take 20-40s, so the real generate call needs a
    # generous timeout -- but that same generous timeout would make a
    # deployment with no Ollama at all (e.g. Render) hang pointlessly before
    # falling back. Ping a cheap endpoint with a short timeout first to tell
    # "not running" apart from "running but slow to generate."
    try:
        with urllib.request.urlopen(OLLAMA_PING_URL, timeout=OLLAMA_PING_TIMEOUT):
            return True
    except Exception:
        return False


def _call_ollama(prompt: str) -> str:
    if not _ollama_available():
        raise RuntimeError("Ollama is not reachable at " + OLLAMA_HOST)
    # num_predict caps generation length -- 3 short sentences never need more
    # than ~120 tokens, and capping it keeps a slow CPU decode from running on.
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 120},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_GENERATE_TIMEOUT) as resp:
        body = json.loads(resp.read())
    return body["response"].strip()


def _call_hosted_gemma(prompt: str) -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("no GEMINI_API_KEY configured for the hosted-Gemma fallback")
    client = genai.Client(api_key=api_key)

    last_exc = None
    for attempt in range(3):  # the hosted API intermittently 503/500s under load
        try:
            resp = client.models.generate_content(model=HOSTED_GEMMA_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise last_exc


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
        backend = f"ollama-local:{MODEL}"
    except Exception as ollama_exc:
        try:
            text = _call_hosted_gemma(prompt)
            backend = f"gemini-hosted:{HOSTED_GEMMA_MODEL}"
        except Exception as hosted_exc:  # pragma: no cover - environment dependent
            verb = "above" if direction == "spike" else "below"
            text = (
                f"Sales were {finding['shortfall_pct']:.0f}% {verb} expected because {CAUSE_LABELS.get(finding['cause'], finding['cause'])}."
            )
            backend = f"fallback-template (ollama: {ollama_exc!r}, hosted: {hosted_exc!r})"
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
