"""CPU-only проверки золотого сечения, точного окна, журнала и защит возобновления."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from scripts.dev_1_4_server_preparation import profile_golden_run as runner
from scripts.dev_1_4_server_preparation.check_surface_runner import expect_error
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import number, read_json


def checks() -> None:
    root = runner.ROOT / "cpu_checks_v1"
    root.mkdir(parents=True, exist_ok=False)
    calls: list[float] = []

    def quadratic(request: runner.Request) -> float:
        calls.append(request.factor)
        return (request.factor-.63)**2+.1

    result = runner.golden_search(quadratic)
    assert len(calls) == 9 and len(set(calls)) == 9 and len(result.history) == 8
    assert result.final_interval[0] <= .63 <= result.final_interval[1]
    assert result.final_interval[1]-result.final_interval[0] < .02
    assert calls[:2] == [1.-runner.RATIO*.75, .25+runner.RATIO*.75]
    for entry in result.history:
        before = entry["before_interval"]
        assert isinstance(before, list)
        np.testing.assert_allclose(number(entry["width"]), (before[1]-before[0])*runner.RATIO, rtol=1e-13)
    replay_calls: list[float] = []

    def replay_quadratic(request: runner.Request) -> float:
        replay_calls.append(request.factor)
        return (request.factor-.63)**2+.1

    replay = runner.golden_search(replay_quadratic)
    assert asdict(replay) == asdict(result) and replay_calls == calls
    flat = runner.golden_search(lambda request: 1.)
    assert len(flat.evaluations) == 9 and all(event["branch"] == "keep_left" for event in flat.history)
    decreasing = runner.golden_search(lambda request: 2.-request.factor)
    assert decreasing.final_interval[1] == 1. and len(decreasing.evaluations) == 9
    aborted: list[int] = []

    def fail_third(request: runner.Request) -> float:
        aborted.append(request.index)
        if request.index == 3:
            raise RuntimeError("intentional CPU abort")
        return (request.factor-.63)**2

    expect_error(lambda: runner.golden_search(fail_third), RuntimeError)
    assert aborted == [1, 2, 3]
    limited: list[int] = []

    def limited_value(request: runner.Request) -> float:
        limited.append(request.index)
        return request.factor

    expect_error(lambda: runner.golden_search(limited_value,
        runner.SearchConfig(max_evaluations=2)), RuntimeError)
    assert limited == [1, 2]
    expect_error(lambda: runner.golden_search(lambda request: float("nan")), ValueError)
    times = runner.inclusive_times(1., .001)
    values = np.ones((1001, 70, 2), dtype=np.float64)
    values[:500] = 1e6
    assert runner.objective(values, times) == np.sqrt(501*70*2)
    expect_error(lambda: runner.objective(values[:1000], times[:1000]), ValueError)
    prefix = runner.source_prefix_evidence()
    assert prefix["prefix_array_equal"] is True and prefix["historical_2s_sha256"] != prefix["new_1s_sha256"]
    controls = runner.controls()
    assert [entry["factor"] for entry in controls] == [1., .5, .25]
    series = root / "mock_series"
    series.mkdir()
    manifest: dict[str, object] = {"controls": [{"objective": 2.}], "scope": "CPU MOCK ONLY"}
    executed: list[int] = []

    def mock_execute(request: runner.Request, supplied: dict[str, object], output: Path) -> dict[str, object]:
        assert supplied == manifest
        executed.append(request.index)
        output.mkdir(parents=True, exist_ok=False)
        mock_values = np.zeros((1001, 70, 2), dtype=np.float64)
        mock_values[500:] = (request.factor-.63)**2+.1
        with (output / "surface.npz").open("xb") as stream:
            np.savez(stream, time_s=times, coordinates_m=np.column_stack((np.arange(5., 700., 10.), np.zeros(70))),
                node_ids=np.arange(70, dtype=np.int64), velocity_m_s=mock_values)
        metadata: dict[str, object] = {"status": "CPU MOCK ONLY", "surface_file": file_record(output / "surface.npz"),
            "wall_seconds": 0., "request": asdict(request)}
        runner.exclusive_json(output / "metadata.json", metadata)
        runner.exclusive_json(output / "validation.json", {"status": "passed", "scope": "CPU MOCK ONLY", "metadata": file_record(output / "metadata.json")})
        return metadata

    def mock_verify(output: Path, request: runner.Request, supplied: dict[str, object]) -> dict[str, object]:
        assert supplied == manifest
        metadata = read_json(output / "metadata.json")
        assert metadata["request"] == asdict(request)
        return metadata

    with patch.object(runner, "verify_manifest", return_value=manifest), patch.object(runner, "execute_run", side_effect=mock_execute), patch.object(runner, "verify_run", side_effect=mock_verify):
        session = runner.Session(series, manifest)
        request = runner.Request(1, calls[0], .25, 1., "initial_left")
        first = session.evaluate(request)
        resumed = runner.Session(series, manifest)
        assert resumed.evaluate(request) == first and executed == [1]
        runner.audit_existing(series)
        ledger = series / "evaluations/eval_01.json"
        original = ledger.read_text()
        ledger.write_text(original.replace('"objective": ', '"objective": 9e9, "original_objective": ', 1))
        expect_error(lambda: runner.Session(series, manifest).evaluate(request), ValueError)
        ledger.write_text(original)
        request2 = runner.Request(2, calls[1], .25, 1., "initial_right")
        (series / "runs/eval_02").mkdir()
        expect_error(lambda: runner.Session(series, manifest).evaluate(request2), ValueError)
        assert executed == [1]
    with runner.session_lock(series):
        expect_error(lambda: enter_lock(series), RuntimeError)
    (series / "requests/eval_04.json").write_text("{}")
    expect_error(lambda: runner.audit_existing(series), ValueError)
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    runner.exclusive_json(root / "validation.json", {"status": "passed", "utc": runner.now(),
        "scope": "CPU mock only; no native/GPU calculation", "code": runner.code_records(), "source_prefix": prefix,
        "checks": ["Quadratic minimum retained; deterministic replay;9 unique evaluations and8 contractions; no final extra solve",
            "Flat objective deterministic tie rule; monotone boundary case; abort/budget/nonfinite guards",
            "Objective includes exactly501x70x2 samples at0.5–1s; early0–0.5s excluded",
            "Historical2s source SHA reconstructed; first96001 samples exactly equal new1s force table",
            "Existing evaluated result reused without a solver call; ledger tamper and incomplete run rejected",
            "Exclusive session lock and orphan-record audit"], "controls": controls})
    print("Golden CPU checks passed; no native/GPU loaded", flush=True)


def enter_lock(root: Path) -> None:
    with runner.session_lock(root):
        pass


if __name__ == "__main__":
    checks()
