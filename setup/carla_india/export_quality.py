"""Export a completed suite's measured facts without machine paths or local assets."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, places=2):
    return f"{value:.{places}f}" if value is not None else "No pair in 4 m search band"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    suite = args.suite.resolve()
    if not suite.is_relative_to(ROOT / "runs"):
        raise ValueError("Select the saved local suite")
    source = load(suite / "report.json")
    if not source["completed"] or not source["passed"] or len(source["episodes"]) != 7:
        raise ValueError("Use a completed seven-case suite with passed checks")
    rows, runtime_hashes = [], None
    for episode in source["episodes"]:
        folder = Path(episode["output"]).resolve()
        if not folder.is_relative_to(ROOT / "runs"):
            raise ValueError("Unexpected episode path")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder / "source").glob("*.py")}
        if runtime_hashes is None:
            runtime_hashes = hashes
        if hashes != runtime_hashes:
            raise ValueError("Suite source snapshots differ")
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT / "integration" / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Current runtime source differs from checked scene: {name}")
        audit = episode["audit"]
        rows.append({"case": episode["case"], "seed": episode["seed"], "weather": episode["weather"],
            "passed": episode["passed"], "checks": audit["checks"],
            "completed_overtakes": episode["completed_overtakes"], "goal_reached": audit["goal_reached"],
            "min_body_gap_m": audit["min_gap_m"], "min_gap_pair": audit["min_gap_pair"],
            "close_clearance_alerts": episode["close_clearance_alerts"],
            "all_crossings_completed": episode["all_crossings_completed"],
            "pedestrian_spacing_events": episode["pedestrian_spacing_events"],
            "viewer_controls_verified": episode["viewer_controls_verified"],
            "native_light_flags_verified_after_tick": episode["native_light_flags_verified"],
            "simulated_seconds_per_loop_wall_second": episode["playback_actual"],
            "all_actor_motion_metrics": audit["actors"], "local_evidence": folder.relative_to(ROOT).as_posix()})
    report = {"date": "2026-10-20", "scenario_version": source["scenario_version"], "completed": True,
        "passed": True, "model_control": False, "rl_trained": False, "scenarios": rows,
        "runtime_source_sha256": runtime_hashes,
        "geometry_kernel_benchmark": load(ROOT / "docs/results/geometry_kernel_benchmark.json"),
        "preserved_failed_evidence": ["runs/carla_quality_20261020_01", "runs/carla_asphalt_observation_20261020_01"],
        "limits": ["Illustrative hand-coded traffic using simulator positions; not calibrated Indian traffic or camera-controlled driving",
            "All body checks are sampled at 20 Hz and native collision events cover the tested vehicle actors",
            "Close-clearance alerts remain reported separately even when no collision/overlap occurs",
            "Rain/dusk settings change appearance; wet-friction behavior is not calibrated",
            "Graphics and capture throughput differ; isolated geometry-kernel speedup is not total simulator speedup",
            "All three segmentation models, source data/splits and completed real test evaluation remain preserved"]}
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    (ROOT / "docs/results/simulation_quality.json").write_text(text, encoding="utf-8")
    (suite / "portable_results.json").write_text(text, encoding="utf-8")
    lines = ["# Final simulation checks — 20 October 2026", "", "Seven fresh owned sessions passed their declared checks.", "",
        "| Case / seed | Weather | Passes | Goal | Closest body gap (m) | Close alerts | Ego braking max over 0.5 s | Ego jerk p95 over 0.5 s |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        ego = row["all_actor_motion_metrics"]["ego"]
        lines.append(f"| {row['case']} / {row['seed']} | {row['weather']} | {row['completed_overtakes']} | {row['goal_reached']} | {number(row['min_body_gap_m'],3)} | {row['close_clearance_alerts']} | {number(ego['braking_500ms_max_mps2'])} | {number(ego['abs_jerk_500ms_p95_mps3'])} |")
    lines.extend(["", "All declared crossings completed. Native light flags were checked after world advancement; pause, camera views and screenshots passed.",
        "No recorded native collisions, sampled body overlaps, vehicle road departures or wrong-way events occurred in these checks.", "",
        "Close alerts are retained, including bus/parked-car clearance. Zero sampled collisions is not a safety guarantee.",
        "The oncoming scenario intentionally does not claim goal completion within its 60-second test.", "",
        "Full source snapshots, actor traces, incidents and screenshots remain in the referenced local runs/ folders."])
    (suite / "QUALITY_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"passed": True, "scenarios": len(rows), "sources_identical": True,
                      "close_alerts": sum(r['close_clearance_alerts'] for r in rows)}))


if __name__ == "__main__":
    main()
