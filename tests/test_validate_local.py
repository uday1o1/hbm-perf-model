import json

import numpy as np
import pytest

from hbmperf.validate import local

FILES = {
    "calibration": [("qwen2.5-0.5b", "q8_0"), ("qwen2.5-0.5b", "f16"), ("qwen2.5-1.5b", "q8_0"),
                    ("qwen2.5-1.5b", "f16")],
    "evaluation": [("smollm2-360m", "q8_0"), ("smollm2-360m", "f16"), ("smollm2-1.7b", "q8_0"),
                   ("smollm2-1.7b", "f16"), ("tinyllama-1.1b", "q8_0"),
                   ("tinyllama-1.1b", "f16")],
}
DEPTHS = [0, 1024, 2048, 4096]


def _session(tmp, split, eta=0.6, t0=2e-3, noise=0.03, anchor_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    records, files = [], {}
    for model_id, prec in FILES[split]:
        name = f"{model_id}-{prec}.gguf"
        size = local.unit_memory_time({"model_id": model_id, "precision": prec, "depth": 0,
                                       "n_gen": 0}) * 100e9
        files[name] = {"file_tensor_bytes": int(size * 1.2), "guard_b_pass": True}
        for d in DEPTHS:
            base = {"filename": name, "model_id": model_id, "precision": prec, "depth": d,
                    "n_gen": 128}
            t = local.unit_memory_time(base) / eta + t0
            records.append({**base, "status": "ok",
                            "step_time_s": t * (1 + noise * rng.standard_normal())})
    anchor = {"status": "ok", "step_time_s": 0.01 * anchor_scale}
    s = {"records": records, "files": files, "anchors": [anchor, anchor]}
    (tmp / f"{split}-x.json").write_text(json.dumps(s))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "CAL_PATH", tmp_path / "cal.json")
    monkeypatch.setattr(local, "REPORT_PATH", tmp_path / "c1.json")
    return tmp_path


def test_recovers_calibration_and_passes(paths):
    _session(paths, "calibration")
    _session(paths, "evaluation", seed=1)
    cal = local.calibrate(paths)
    assert cal["candidate"]["eta_m"] == pytest.approx(0.6, rel=0.1)
    rep = local.evaluate(paths)
    assert rep["verdict"] == "PASS", rep["reasons"]
    assert rep["comparison"]["mdape"] < 0.05
    assert rep["spec_peak_mdape"] > 0.3
    assert local.status(paths) == "CURRENT"


def test_anchor_drift_gives_insufficient_evidence(paths):
    _session(paths, "calibration")
    _session(paths, "evaluation", seed=1, anchor_scale=1.3)
    local.calibrate(paths)
    rep = local.evaluate(paths)
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "drift_normalized_mdape" in rep


def test_wrong_model_fails(paths):
    _session(paths, "calibration")
    _session(paths, "evaluation", seed=1, eta=0.3)  # evaluation hardware behaves differently
    local.calibrate(paths)
    assert local.evaluate(paths)["verdict"] == "FAIL"


def test_editing_a_measurement_makes_report_stale(paths):
    _session(paths, "calibration")
    _session(paths, "evaluation", seed=1)
    local.calibrate(paths)
    local.evaluate(paths)
    f = paths / "evaluation-x.json"
    data = json.loads(f.read_text())
    data["records"][0]["step_time_s"] *= 1.01
    f.write_text(json.dumps(data))
    assert local.status(paths) == "STALE"
