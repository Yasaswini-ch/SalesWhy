"""FastAPI backend for SalesWhy. Serves static/index.html and three endpoints:
upload+analyze, sample-shop analyze, and re-explain in a different language.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402
from explain import explain  # noqa: E402
from drilldown import DrilldownError  # noqa: E402
from mapper import guess_mapping  # noqa: E402
from slip import render_slip  # noqa: E402
from whatif import WhatIfError  # noqa: E402

app = FastAPI(title="SalesWhy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAMPLE_CSV = ROOT / "data" / "sample_shop.csv"


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/sample-columns")
def sample_columns():
    """Lets the UI show the auto-mapper guess for the sample shop without a real upload."""
    df = pd.read_csv(SAMPLE_CSV, nrows=200)
    return {"columns": list(df.columns), "mapping": guess_mapping(df)}


@app.post("/api/mapping")
async def mapping(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=200)
    except Exception:
        raise HTTPException(400, "Couldn't read this as a CSV file.")
    return {"columns": list(df.columns), "mapping": guess_mapping(df)}


def _run_pipeline(df: pd.DataFrame, mapping_override: dict | None, language: str) -> dict:
    try:
        return pipeline.run(df, mapping=mapping_override, language=language)
    except pipeline.PipelineError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    date_col: str | None = Form(None),
    amount_col: str | None = Form(None),
    quantity_col: str | None = Form(None),
    product_col: str | None = Form(None),
    language: str = Form("en"),
):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Couldn't read this as a CSV file.")

    mapping_override = None
    if date_col and amount_col:
        mapping_override = {
            "date": date_col, "amount": amount_col,
            "quantity": quantity_col or None, "product": product_col or None,
        }

    result = _run_pipeline(df, mapping_override, language)
    return JSONResponse(result)


@app.post("/api/analyze-sample")
async def analyze_sample(language: str = Form("en")):
    df = pd.read_csv(SAMPLE_CSV)
    result = _run_pipeline(df, None, language)
    return JSONResponse(result)


class ExplainRequest(BaseModel):
    finding: dict
    language: str = "en"


@app.post("/api/explain")
async def re_explain(req: ExplainRequest):
    """Re-explain a single already-detected finding in a different language."""
    return JSONResponse(explain(req.finding, language=req.language))


class SlipRequest(BaseModel):
    finding: dict
    explanation_text: str
    shop_name: str = "Your Shop"
    backend_footer: str = ""
    language: str = "en"


@app.post("/api/slip")
async def slip(req: SlipRequest):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        render_slip(req.finding, req.explanation_text, req.shop_name, tmp.name, req.backend_footer, req.language)
        return FileResponse(tmp.name, media_type="image/png", filename="saleswhy_ledger_slip.png")


async def _load_source_df(use_sample: bool, file: UploadFile | None) -> pd.DataFrame:
    if use_sample:
        return pd.read_csv(SAMPLE_CSV)
    if file is None:
        raise HTTPException(400, "No file provided.")
    content = await file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Couldn't read this as a CSV file.")


@app.post("/api/whatif")
async def whatif(
    finding_json: str = Form(...),
    date_col: str = Form(...),
    amount_col: str = Form(...),
    quantity_col: str | None = Form(None),
    product_col: str | None = Form(None),
    use_sample: bool = Form(False),
    file: UploadFile | None = File(None),
):
    finding = json.loads(finding_json)
    mapping = {"date": date_col, "amount": amount_col, "quantity": quantity_col or None, "product": product_col or None}
    df = await _load_source_df(use_sample, file)

    try:
        result = pipeline.whatif(df, mapping, finding)
    except (pipeline.PipelineError, WhatIfError) as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse(result)


@app.post("/api/drilldown")
async def drilldown(
    finding_json: str = Form(...),
    date_col: str = Form(...),
    amount_col: str = Form(...),
    quantity_col: str | None = Form(None),
    product_col: str | None = Form(None),
    use_sample: bool = Form(False),
    file: UploadFile | None = File(None),
):
    finding = json.loads(finding_json)
    mapping = {"date": date_col, "amount": amount_col, "quantity": quantity_col or None, "product": product_col or None}
    df = await _load_source_df(use_sample, file)

    try:
        result = pipeline.drilldown(df, mapping, finding)
    except (pipeline.PipelineError, DrilldownError) as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"products": result})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
