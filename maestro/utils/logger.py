import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("runs")
LOG_DIR.mkdir(exist_ok=True)

def log_run(stage: str, data):
    # Use Windows-safe timestamp format
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file = LOG_DIR / f"{stage}_{timestamp}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
