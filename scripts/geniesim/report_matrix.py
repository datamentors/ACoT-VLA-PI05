#!/usr/bin/env python3
"""Summarise a run_sorting_matrix.sh output directory as Markdown (stdout) + JSON.

    python3 report_matrix.py ~/g2sim/runs/matrix_20261008 [--runs ~/g2sim/genie_sim/runs]

Per variant: Genie Sim sub-task successes (each STEPS entry of evaluate_ret), E2E,
episode duration, and from the traces the timing that explains them: inference
latency, rows executed per chunk, seam size, starved/held ticks, and how far the
first command of each episode lands from the previous episode's last one.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def scores(eval_path: pathlib.Path) -> dict | None:
    if not eval_path.exists():
        return None
    eps = json.loads(eval_path.read_text())["details"]
    steps = collections.defaultdict(lambda: [0.0, 0])
    e2e = 0
    durations = []
    any_pick = 0
    for e in eps:
        s = e["result"]["scores"]
        e2e += int(s["E2E"])
        durations.append(float(e["duration"]))
        picked = False
        for st in s["STEPS"]:
            steps[st["name"]][0] += float(st["score"])
            steps[st["name"]][1] += 1
            picked |= st["name"].startswith("PickUp") and st["score"] > 0
        any_pick += picked
    partial = [
        np.mean([x["score"] for x in e["result"]["scores"]["STEPS"]]) if e["result"]["scores"]["STEPS"] else 0.0
        for e in eps
    ]
    return {
        "episodes": len(eps),
        "e2e": e2e,
        "steps": {k: {"success": v[0], "attempts": v[1]} for k, v in steps.items()},
        "episodes_with_pickup": any_pick,
        "mean_partial": float(np.mean(partial)) if partial else 0.0,
        "duration_p50_s": _pct(durations, 50),
    }


_STAGES = ("follow", "pickup", "placed", "pickup2", "placed2")


def stage_times(log_path: pathlib.Path) -> dict | None:
    """Seconds per task stage, from the sim log's StepOut events.

    Each episode starts at "[local_recorder] started episode"; every "StepOut evt: 4"
    (wall-clock seconds, so the stream client, which renders every step, runs slower)
    is a stage completed (1 Follow, 2 first pickup, 3 first package placed inside and
    upright, 4/5 second package), "evt: 3" is the episode ending on a timeout.
    Returns per-stage medians of the time since the previous stage, over episodes that
    completed it, and the median time spent after the last completed stage.
    """
    if not log_path.exists():
        return None
    import datetime as dt

    def ts(line: str) -> float:
        return dt.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()

    episodes = []
    for line in log_path.read_text(errors="replace").splitlines():
        # Only the [logger.py:76] copy: the RTC launchers also echo a ColoredLogger copy.
        if "[logger.py:76] [local_recorder] started episode" in line:
            episodes.append({"start": ts(line), "done": [], "end": None})
        elif episodes and "[logger.py:76] Action [StepOut] evt: 4" in line:
            episodes[-1]["done"].append(ts(line))
        elif episodes and "[logger.py:76] Action [StepOut] evt: 3" in line:
            episodes[-1]["end"] = ts(line)
    per_stage = collections.defaultdict(list)
    stuck = []
    for e in episodes:
        prev = e["start"]
        for name, t in zip(_STAGES, e["done"]):
            per_stage[name].append(t - prev)
            prev = t
        if e["end"] is not None:
            stuck.append(e["end"] - prev)
    out = {f"{k}_s_p50": _pct(per_stage[k], 50) for k in _STAGES if per_stage[k]}
    out.update({f"{k}_n": len(per_stage[k]) for k in _STAGES if per_stage[k]})
    out["after_last_stage_s_p50"] = _pct(stuck, 50)
    return out


def stream_timing(server: list[dict], client: list[dict]) -> dict:
    steady = [x for x in server if x.get("phase") == "steady"]
    seams = [x["seam_arm_max"] for x in steady if x.get("seam_arm_max") is not None]
    cmds = [x for x in client if x.get("type") == "cmd"]
    starts = _episode_start_jumps(client)
    return {
        "inferences": len(server),
        "model_ms_p50": _pct([x["model_ms"] for x in steady if x.get("model_ms") is not None], 50),
        "rows_per_chunk_p50": _pct([30 - x["prev_rows"] for x in steady], 50),
        "consumed_during_inference": dict(collections.Counter(x["consumed"] for x in steady).most_common(4)),
        "seam_rad_p50": _pct(seams, 50),
        "seam_rad_p99": _pct(seams, 99),
        "readiness_rejections": sum(1 for x in server if x.get("phase") == "readiness" and not x.get("sustainable")),
        "acts": len(client),
        "held_ticks": len(client) - len(cmds),
        "latched_replies": sum(1 for x in client if x.get("state") == "latched"),
        "episode_start_jump_rad_p50": _pct(starts, 50),
        "episode_start_jump_rad_max": max(starts) if starts else None,
    }


def _episode_start_jumps(client: list[dict]) -> list[float]:
    """Largest arm change over the first 10 commands of each episode (a stale seed shows here)."""
    by_ep = collections.defaultdict(list)
    for x in client:
        if x.get("type") == "cmd" and x.get("row") is not None:
            by_ep[x["episode"]].append(np.asarray(x["row"][:14]))
    jumps = []
    for rows in by_ep.values():
        if len(rows) > 10:
            jumps.append(float(np.max(np.abs(np.diff(np.stack(rows[:11]), axis=0)))))
    return jumps


def oldrtc_timing(trace: list[dict]) -> dict:
    rtc = [x for x in trace if x.get("rtt_ms") is not None]
    return {
        "inferences": len(rtc),
        "rtt_ms_p50": _pct([x["rtt_ms"] for x in rtc], 50),
        "rows_consumed_per_inference_p50": _pct([x["consumed"] for x in rtc], 50),
        "seam_rad_p50": _pct([x["seam_arm_max"] for x in rtc if x.get("seam_arm_max") is not None], 50),
        "seam_rad_p99": _pct([x["seam_arm_max"] for x in rtc if x.get("seam_arm_max") is not None], 99),
        "starved_ticks_total": max((x.get("starved_total") or 0) for x in rtc) if rtc else None,
    }


def _ident(name: str) -> tuple[str, str, str]:
    """run id -> (task, model, variant). Ids from run_sorting_matrix.sh have no task/model."""
    parts = name.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "sorting_packages", "manip", f"extra:{name}"


def _kind(variant: str) -> str:
    v = variant.removeprefix("extra:")
    if v.startswith("new"):
        return "new"
    if v.startswith("old"):
        return "old"
    return "pause"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("matrix_dir")
    p.add_argument("--runs", default=str(pathlib.Path.home() / "g2sim/genie_sim/runs"))
    a = p.parse_args()
    out = pathlib.Path(a.matrix_dir).expanduser()
    runs = pathlib.Path(a.runs).expanduser()

    names = []
    for line in (out / "progress.log").read_text().splitlines():
        if " START " in line:
            names.append(line.split(" START ")[1].split(" ")[0])
        elif " REUSE " in line:
            names.append(line.split(" -> ")[1].strip())

    report = {}
    for name in dict.fromkeys(names):
        task, model, variant = _ident(name)
        log = out / f"{name}.log"
        entry = {
            "task": task,
            "model": model,
            "variant": variant,
            "scores": scores(out / f"{name}_eval.json"),
            "stages": stage_times(log if log.exists() else out.parent / f"{name}.log"),
        }
        if _kind(variant) == "new":
            entry["timing"] = stream_timing(_load_jsonl(out / f"{name}_server.jsonl"), _load_jsonl(runs / f"{name}.jsonl"))
        elif _kind(variant) == "old":
            entry["timing"] = oldrtc_timing(_load_jsonl(runs / f"{name}.jsonl"))
        report[name] = entry
    (out / "report.json").write_text(json.dumps(report, indent=2))

    groups = collections.defaultdict(list)
    for name, e in report.items():
        groups[(e["task"], e["model"])].append((name, e))
    for (task, model), items in groups.items():
        print(f"## {task} / {model}\n")
        step_names = sorted({k for _, e in items if e["scores"] for k in e["scores"]["steps"]})
        print("| variant | eps | E2E | " + " | ".join(step_names) + " | mean partial | ep dur p50 s |")
        print("|---" * (len(step_names) + 5) + "|")
        for _, e in items:
            s = e["scores"]
            if not s:
                print(f"| {e['variant']} | — | no score file |" + " |" * (len(step_names) + 2))
                continue
            cells = [
                f"{s['steps'][k]['success']:.0f}/{s['steps'][k]['attempts']}" if k in s["steps"] else "—"
                for k in step_names
            ]
            print(
                f"| {e['variant']} | {s['episodes']} | {s['e2e']}/{s['episodes']} | " + " | ".join(cells)
                + f" | {s['mean_partial']:.3f} | {s['duration_p50_s']:.0f} |"
            )
        print()
        print("| variant | to stage 1 s | stage 1→2 s | stage 2→3 s | stage 3→4 s | after last stage s |")
        print("|---|---|---|---|---|---|")
        for _, e in items:
            st = e.get("stages") or {}

            def cell(key: str, st=st) -> str:
                v = st.get(f"{key}_s_p50")
                return "—" if v is None else f"{v:.1f} (n={st[f'{key}_n']})"

            after = st.get("after_last_stage_s_p50")
            print(
                f"| {e['variant']} | {cell('follow')} | {cell('pickup')} | {cell('placed')} | {cell('pickup2')} | "
                + ("—" if after is None else f"{after:.1f}") + " |"
            )
        print()
        for _, e in items:
            if "timing" in e:
                print(
                    f"- **{e['variant']}** timing: "
                    + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e["timing"].items())
                )
        print()


if __name__ == "__main__":
    main()
