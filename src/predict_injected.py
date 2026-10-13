"""Run predict.py's model over the injected (synthetic-drop) feature set.

Separate entry point from predict.py's main() so the real evaluation run
never gets mixed up with the clean run's predictions/backend log.
"""
from pathlib import Path

from predict import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(
        root / "results" / "injected",
        root / "results" / "injected_predictions",
        backend_log_name="predict_backend_log_injected.csv",
    )
