import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
from pathlib import Path

def test_artifacts_exist():
    assert Path("model/model.pkl").exists()
    assert Path("model/metrics.json").exists()

def test_r2_above_threshold():
    metrics = json.loads(Path("model/metrics.json").read_text())
    assert metrics["r2"] > 0.4
