"""Золотое сечение силы квадратичного слоя 175 м: локальные опыты длиной 1 с."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_3_point_source.physics import ExperimentSpec, inclusive_times, ricker
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_4_server_preparation import profile_sweep_run as sweep
from scripts.dev_1_4_server_preparation.surface_runner import _check_write, _write_metadata
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence

ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/profile_golden_1s_3060_v1"
SWEEP_ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/profile_sweep_3060_v1"
RATIO = (math.sqrt(5.) - 1.) / 2.
T_END = 1.
WINDOW = (.5, 1.)
FloatArray = NDArray[np.float64]
INITIAL_STATE = "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3"


@dataclass(frozen=True)
class SearchConfig:
    lower: float = .25
    upper: float = 1.
    tolerance: float = .02
    max_evaluations: int = 9

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.lower, self.upper, self.tolerance)):
            raise ValueError("Nonfinite search configuration")
        if not 0 < self.lower < self.upper <= 1 or not 0 < self.tolerance < self.upper-self.lower or self.max_evaluations < 2:
            raise ValueError("Invalid search configuration")


CONFIG = SearchConfig()


@dataclass(frozen=True)
class Request:
    index: int
    factor: float
    lower: float
    upper: float
    role: str

    @property
    def name(self) -> str:
        return f"eval_{self.index:02d}"


@dataclass(frozen=True)
class SearchResult:
    final_interval: tuple[float, float]
    evaluations: list[dict[str, object]]
    history: list[dict[str, object]]


def golden_search(evaluate: Callable[[Request], float], config: SearchConfig = CONFIG,
                  on_contraction: Callable[[dict[str, object]], None] | None = None) -> SearchResult:
    """Two initial samples, then exactly one new sample per unfinished contraction."""
    a, b = config.lower, config.upper
    evaluations: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    def sample(x: float, role: str) -> tuple[int, float]:
        index = len(evaluations)+1
        if index > config.max_evaluations:
            raise RuntimeError("Evaluation budget exceeded before solver invocation")
        request = Request(index, x, a, b, role)
        value = evaluate(request)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Objective must be finite and nonnegative")
        evaluations.append({**asdict(request), "objective": value})
        return index, value

    left, right = b-RATIO*(b-a), a+RATIO*(b-a)
    left_id, fl = sample(left, "initial_left")
    right_id, fr = sample(right, "initial_right")
    while b-a > config.tolerance:
        old = (a, b)
        compared = {"left": left, "right": right, "left_objective": fl,
                    "right_objective": fr, "compared_ids": [left_id, right_id]}
        if fl <= fr:
            b = right
            right, right_id, fr = left, left_id, fl
            branch = "keep_left"
        else:
            a = left
            left, left_id, fl = right, right_id, fr
            branch = "keep_right"
        event: dict[str, object] = {"iteration": len(history)+1, "before_interval": list(old),
            "lower": a, "upper": b, "width": b-a, "branch": branch, **compared}
        history.append(event)
        if on_contraction is not None:
            on_contraction(event)
        if b-a <= config.tolerance:
            break
        if branch == "keep_left":
            left = b-RATIO*(b-a)
            left_id, fl = sample(left, "new_left")
        else:
            right = a+RATIO*(b-a)
            right_id, fr = sample(right, "new_right")
    return SearchResult((a, b), evaluations, history)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sweep.homogeneous.write_json(path, value)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    """Only the explicitly mutable checkpoint is replaced; immutable records use x."""
    if path.name != "checkpoint.json":
        raise ValueError("Atomic replacement is reserved for checkpoint.json")
    temporary = path.with_name("checkpoint.next.json")
    if temporary.exists():
        raise FileExistsError("Stale checkpoint.next.json requires inspection")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def code_records() -> list[dict[str, object]]:
    paths = [Path(__file__), Path(__file__).with_name("check_profile_golden.py"), Path(sweep.__file__),
        Path(sweep.homogeneous.__file__), Path(sweep.fields.__file__),
        Path(__file__).with_name("surface_runner.py"), PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py",
        PROJECT_ROOT / "scripts/dev_1_2_smoothing/pilot_run.py", PROJECT_ROOT / "scripts/dev_1_2_smoothing/pilot_mesh.py"]
    return [file_record(path) for path in paths]


def source_prefix_evidence() -> dict[str, object]:
    historical = np.zeros((192001, 1, 2), dtype=np.float64)
    historical[:, 0, 1] = -ricker(np.linspace(0., 2., 192001))
    target = np.zeros((96001, 1, 2), dtype=np.float64)
    target[:, 0, 1] = -ricker(np.linspace(0., T_END, 96001))
    np.testing.assert_array_equal(target, historical[:96001])
    old_sha = hashlib.sha256(historical.tobytes()).hexdigest()
    reference = read_json(sweep.REFERENCE / "metadata.json")
    if old_sha != reference["source_input_sha256"]:
        raise ValueError("Reconstructed historical 2s force table does not match its SHA")
    return {"historical_2s_sha256": old_sha, "new_1s_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "historical_samples": 192001, "target_samples": 96001, "prefix_array_equal": True,
        "method": "Reconstruct historical 2s nodal force table, verify full SHA, then compare its first96001 samples exactly to target1s table",
        "full_hashes_equal": False, "downward_force_amplitude_n_per_m": 1.}


def objective(values: FloatArray, times: FloatArray) -> float:
    if values.shape != (len(times), 70, 2) or not np.isfinite(values).all():
        raise ValueError("Invalid velocity samples")
    mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    if np.count_nonzero(mask) != 501 or not np.array_equal(times[mask], inclusive_times(1., .001)[500:]):
        raise ValueError("Objective window must contain exactly501 samples from0.5 to1s")
    return float(np.sqrt(np.sum(values[mask]**2, dtype=np.float64)))


def surface_objective(directory: Path, expected_samples: int) -> float:
    meta = read_json(directory / "metadata.json")
    if mapping(meta["surface_file"])["sha256"] != file_record(directory / "surface.npz")["sha256"]:
        raise ValueError("Surface SHA mismatch")
    with np.load(directory / "surface.npz", allow_pickle=False) as archive:
        times, coords, ids, values = (archive[key] for key in ("time_s", "coordinates_m", "node_ids", "velocity_m_s"))
        if times.dtype != np.float64 or values.dtype != np.float64 or coords.dtype != np.float64 or ids.dtype != np.int64:
            raise ValueError("Wrong storage dtype")
        if len(times) != expected_samples or not np.array_equal(times, inclusive_times((expected_samples-1)*.001, .001)):
            raise ValueError("Wrong sampling axis")
        if coords.shape != (70, 2) or not np.allclose(coords[:, 0], np.arange(5., 700., 10.), rtol=0, atol=1e-9) or not np.allclose(coords[:, 1], np.zeros(70), rtol=0, atol=1e-9):
            raise ValueError("Wrong receiver coordinates")
        if ids.shape != (70,) or len(np.unique(ids)) != 70 or not np.array_equal(values[0], np.zeros((70, 2))):
            raise ValueError("Wrong receiver IDs or initial velocity")
        return objective(values, times)


def controls() -> list[dict[str, object]]:
    reference = read_json(sweep.REFERENCE / "metadata.json")
    records: list[dict[str, object]] = []
    for name, factor, directory in (("baseline", 1., sweep.REFERENCE),
            ("q2_half", .5, SWEEP_ROOT / "runs/q2_half"), ("q2_quarter", .25, SWEEP_ROOT / "runs/q2_quarter")):
        metadata = read_json(directory / "metadata.json")
        if metadata.get("status") != "completed" or read_json(directory / "validation.json").get("status") != "passed" or (directory / "metadata.failed.json").exists():
            raise ValueError("Unvalidated historical control")
        for key in ("nodes_sha256", "source_input_sha256", "initial_state", "dt_s", "steps_count", "physical_model", "abc_edge_count"):
            if metadata[key] != reference[key]:
                raise ValueError(f"Control identity differs: {key}")
        if mapping(metadata["native"])["sha256"] != mapping(reference["native"])["sha256"] or mapping(metadata["mesh_evidence"])["sha256"] != mapping(reference["mesh_evidence"])["sha256"]:
            raise ValueError("Control native or mesh differs")
        records.append({"name": name, "factor": factor, "directory": str(directory), "objective": surface_objective(directory, 2001),
            "metadata": file_record(directory / "metadata.json"), "surface": file_record(directory / "surface.npz"),
            "sampling": "Existing2s record; objective uses only inclusive0.5–1.0s; old arrays unchanged"})
    return records


def prepare() -> dict[str, object]:
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "manifest.json").exists() or (ROOT / "runs").exists():
        raise FileExistsError("Golden series already prepared or contains runs")
    reference = read_json(sweep.REFERENCE / "metadata.json")
    native = mapping(reference["native"])
    if file_record(Path(str(native["path"])))["sha256"] != native["sha256"]:
        raise ValueError("Installed native changed")
    samples = golden_search(lambda request: (request.factor-.6)**2)
    manifest: dict[str, object] = {"status": "prepared", "utc": now(), "config": asdict(CONFIG),
        "t_end_s": T_END, "time_window_s": list(WINDOW), "objective_definition": "Unweighted L2 of501x70x2 Vx/Vy samples, inclusive0.5–1s; not physical energy or true reflection coefficient",
        "expected_new_evaluations": len(samples.evaluations), "expected_contractions": len(samples.history),
        "code": code_records(), "controls": controls(), "source_prefix": source_prefix_evidence(),
        "identity": {key: reference[key] for key in ("nodes_sha256", "initial_state", "dt_s", "physical_model", "abc_edge_count", "mesh_evidence", "geometry_evidence", "native")},
        "semgpu2": version("semgpu2"), "solver_run": False,
        "scope": "One homogeneous central source, p4/K96, unchanged175m mesh/ABC; only damping amplitude varies; localRTX3060",
        "selection": "Best observed objective including historical controls; final golden interval and best new point reported separately; golden search assumes unimodality"}
    exclusive_json(ROOT / "manifest.json", manifest)
    return manifest


def verify_manifest(root: Path = ROOT) -> dict[str, object]:
    manifest = read_json(root / "manifest.json")
    if manifest["config"] != asdict(CONFIG) or manifest["t_end_s"] != T_END or manifest["time_window_s"] != list(WINDOW):
        raise ValueError("Search specification changed")
    if manifest["code"] != code_records() or manifest["semgpu2"] != version("semgpu2"):
        raise ValueError("Code or package changed since preparation")
    if manifest["source_prefix"] != source_prefix_evidence() or manifest["controls"] != controls():
        raise ValueError("Historical controls/source identity changed")
    native = mapping(mapping(manifest["identity"])["native"])
    if file_record(Path(str(native["path"])))["sha256"] != native["sha256"]:
        raise ValueError("Native changed since preparation")
    return manifest


def verify_run(output: Path, request: Request, manifest: dict[str, object]) -> dict[str, object]:
    profile = sweep.Profile(request.name, "power", 2., request.factor)
    metadata = sweep.verify(output, profile, T_END)
    identity = mapping(manifest["identity"])
    for key in ("nodes_sha256", "initial_state", "dt_s", "physical_model", "abc_edge_count"):
        if metadata[key] != identity[key]:
            raise ValueError(f"New run identity mismatch: {key}")
    for key in ("native", "mesh_evidence", "geometry_evidence"):
        if mapping(metadata[key])["sha256"] != mapping(identity[key])["sha256"]:
            raise ValueError(f"New run evidence mismatch: {key}")
    if metadata.get("golden_request") != asdict(request) or metadata.get("source_prefix") != manifest["source_prefix"]:
        raise ValueError("Wrong exact factor/request/source-prefix evidence")
    if metadata["source_input_sha256"] != mapping(manifest["source_prefix"])["new_1s_sha256"]:
        raise ValueError("New source table differs from validated historical1s prefix")
    if mapping(metadata["profile_stability"])["positive_nodal_increment_over_reference_s_inv"] != 0.:
        raise ValueError("New profile exceeds conservative reference damping")
    if metadata["steps_count"] != 96000 or metadata["recorded_counts"] != {"surface": 1001, "fields": 0, "source_history": 0}:
        raise ValueError("Wrong1s steps/output count")
    return metadata


def execute_run(request: Request, manifest: dict[str, object], output: Path) -> dict[str, object]:
    """New output wrapper; frozen sweep functions and its ROOT remain unchanged."""
    output.resolve().relative_to(ROOT / "runs")
    if output.exists():
        raise FileExistsError(output)
    profile = sweep.Profile(request.name, "power", 2., request.factor)
    spec = sweep.spec_for(profile, T_END)
    case, edges = sweep.homogeneous.load_case("padded", spec)
    mesh, geometry, certificate = sweep.homogeneous.mesh_inputs("padded")
    stability = sweep.profile_stability(case, profile)
    _check_write(output.parent, 10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": "padded", "padding_m": 175.,
        "profile": sweep.profile_record(profile), "material_model": sweep.homogeneous.MATERIAL_MODEL,
        "physical_model": sweep.homogeneous.PHYSICAL_MODEL, "is_original_dataset_model": False,
        "run_spec": asdict(spec), "experiment_spec": asdict(ExperimentSpec(end_s=T_END)), "golden_request": asdict(request),
        "source_prefix": manifest["source_prefix"], "boundary_mode": "outer_abc", "mass_damping_disabled": False,
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata, "mesh_evidence": file_record(mesh),
        "geometry_evidence": file_record(geometry), "reference_stability_evidence": file_record(certificate),
        "profile_stability": stability, "short_test": False, "components": ["Vx", "Vy"], "receiver_count": 70,
        "save_fields": False, "field_components": [], "field_cadence_s": None, "field_shape": None,
        "field_node_order": None, "initial_state": INITIAL_STATE,
        "timing_note": "Golden search:1s surface-only; no all-node output",
        "provenance": {"python": sys.executable, "numpy": np.__version__, "semgpu2": version("semgpu2"), "files": code_records()}}
    try:
        metadata.update(sweep.execute(case, spec, output, metadata, edges, profile, False))
        metadata.update(status="completed", wall_seconds=perf_counter()-started)
        _write_metadata(output / "metadata.json", metadata, None)
        verified = verify_run(output, request, manifest)
        exclusive_json(output / "validation.json", {"status": "passed", "metadata": file_record(output / "metadata.json"),
            "source_prefix_verified": True, "recorded_counts": metadata["recorded_counts"],
            "scope": "Complete finite1s output and exact physical/source-prefix identity; not independent solver validation"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter()-started, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise


class Session:
    def __init__(self, root: Path, manifest: dict[str, object]) -> None:
        self.root, self.manifest = root, manifest
        self.evaluations: list[dict[str, object]] = []
        self.history: list[dict[str, object]] = []

    def checkpoint(self, status: str, pending: Request | None = None) -> None:
        atomic_json(self.root / "checkpoint.json", {"status": status, "utc": now(), "config": asdict(CONFIG),
            "t_end_s": T_END, "time_window_s": list(WINDOW), "evaluations": self.evaluations,
            "history": self.history, "pending": asdict(pending) if pending else None})

    def contract(self, event: dict[str, object]) -> None:
        self.history.append(event)
        self.checkpoint("searching")

    def evaluate(self, request: Request) -> float:
        verify_manifest(self.root)
        directory = self.root / "runs" / request.name
        ledger = self.root / "evaluations" / f"{request.name}.json"
        requested = self.root / "requests" / f"{request.name}.json"
        if requested.exists():
            if read_json(requested) != asdict(request):
                raise ValueError("Stored request does not match deterministic replay")
        else:
            exclusive_json(requested, asdict(request))
        self.checkpoint("evaluating", request)
        if (directory / "metadata.failed.json").exists():
            raise ValueError("Failed result requires manual inspection; automatic retry forbidden")
        if directory.exists():
            if not (directory / "metadata.json").exists() or not (directory / "validation.json").exists():
                raise ValueError("Incomplete existing result; automatic overwrite/retry forbidden")
            metadata = verify_run(directory, request, self.manifest)
        else:
            if ledger.exists():
                raise ValueError("Evaluation record exists but its numerical result is missing")
            metadata = execute_run(request, self.manifest, directory)
        validation = read_json(directory / "validation.json")
        if validation.get("status") != "passed" or validation.get("metadata") != file_record(directory / "metadata.json"):
            raise ValueError("Invalid result validation")
        value = surface_objective(directory, 1001)
        baseline = number(mapping(sequence(self.manifest["controls"])[0])["objective"])
        if baseline <= 0:
            raise ValueError("Control objective must be positive")
        record: dict[str, object] = {**asdict(request), "directory": str(directory), "objective": value,
            "relative_baseline": value/baseline, "wall_seconds": metadata["wall_seconds"],
            "metadata": file_record(directory / "metadata.json"), "surface": file_record(directory / "surface.npz"),
            "validation": file_record(directory / "validation.json")}
        if ledger.exists():
            if read_json(ledger) != record:
                raise ValueError("Stored evaluation differs from revalidated result")
        else:
            exclusive_json(ledger, record)
        self.evaluations.append(record)
        self.checkpoint("evaluated")
        print(json.dumps({"stage": "evaluated", "index": request.index, "factor": request.factor,
            "objective": value, "relative_baseline": value/baseline, "wall_seconds": metadata["wall_seconds"]}), flush=True)
        return value


@contextmanager
def session_lock(root: Path) -> Iterator[None]:
    with (root / "runner.lock").open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another golden session already holds the lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def audit_existing(root: Path) -> None:
    groups: dict[str, set[str]] = {}
    allowed = {f"eval_{index:02d}" for index in range(1, CONFIG.max_evaluations+1)}
    for name in ("runs", "evaluations", "requests"):
        directory = root / name
        groups[name] = set()
        if directory.exists():
            for path in directory.iterdir():
                if (name == "runs" and not path.is_dir()) or (name != "runs" and (not path.is_file() or path.suffix != ".json")):
                    raise ValueError(f"Unexpected entry: {path}")
                groups[name].add(path.name if name == "runs" else path.stem)
        if not groups[name] <= allowed:
            raise ValueError("Unknown evaluation IDs; refusing any new solver run")
    requested = groups["requests"]
    if requested != {f"eval_{index:02d}" for index in range(1, len(requested)+1)}:
        raise ValueError("Noncontiguous requested evaluations")
    if not groups["evaluations"] <= groups["runs"] <= requested:
        raise ValueError("Orphaned evaluation/run/request records")


def run_session(resume: bool) -> dict[str, object]:
    manifest = verify_manifest()
    with session_lock(ROOT):
        if (ROOT / "failure.json").exists():
            raise ValueError("Failure evidence exists; inspect manually before any retry")
        if not resume and any((ROOT / name).exists() for name in ("checkpoint.json", "runs", "evaluations", "requests", "result.json")):
            raise FileExistsError("Use explicit resume for an existing session")
        audit_existing(ROOT)
        session = Session(ROOT, manifest)
        try:
            result = golden_search(session.evaluate, CONFIG, session.contract)
            expected = {f"eval_{index:02d}" for index in range(1, len(session.evaluations)+1)}
            for name in ("runs", "evaluations", "requests"):
                actual = {path.stem if path.is_file() else path.name for path in (ROOT / name).iterdir()}
                if actual != expected:
                    raise ValueError(f"Unexpected/orphaned records in {name}")
            best_new = min(session.evaluations, key=lambda row: number(row["objective"]))
            observed = [*session.evaluations, *[mapping(row) for row in sequence(manifest["controls"])]]
            best_observed = min(observed, key=lambda row: number(row["objective"]))
            final: dict[str, object] = {"status": "completed", "config": asdict(CONFIG), "t_end_s": T_END,
                "time_window_s": list(WINDOW), "manifest": file_record(ROOT / "manifest.json"),
                "evaluations": session.evaluations, "history": result.history,
                "final_interval": list(result.final_interval), "final_width": result.final_interval[1]-result.final_interval[0],
                "tolerance": CONFIG.tolerance, "best_new": best_new, "best_observed": best_observed,
                "selected_factor": best_observed["factor"], "new_evaluations": len(session.evaluations),
                "bracket_midpoint_not_simulated": sum(result.final_interval)/2,
                "selection_scope": "Minimum observed L2 for this1s homogeneous test; unimodality assumption and finite interval do not certify a global optimum"}
            if (ROOT / "result.json").exists():
                if read_json(ROOT / "result.json") != final:
                    raise ValueError("Completed result differs from deterministic replay")
            else:
                exclusive_json(ROOT / "result.json", final)
            session.checkpoint("completed")
            return final
        except BaseException as error:
            if not (ROOT / "failure.json").exists():
                exclusive_json(ROOT / "failure.json", {"status": "failed", "utc": now(),
                    "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                    "completed_evaluations": len(session.evaluations)})
            session.checkpoint("failed")
            raise


def configure_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        path = ROOT / "runtime_cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "resume", "status"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
        print(json.dumps({"status": result["status"], "expected_new_evaluations": result["expected_new_evaluations"], "root": str(ROOT)}), flush=True)
    elif args.command == "status":
        print(json.dumps(read_json(ROOT / "checkpoint.json") if (ROOT / "checkpoint.json").exists() else {"status": "not_started"}), flush=True)
    else:
        configure_environment()
        result = run_session(args.command == "resume")
        print(json.dumps({"status": result["status"], "selected_factor": result["selected_factor"], "final_interval": result["final_interval"]}), flush=True)


if __name__ == "__main__":
    main()
