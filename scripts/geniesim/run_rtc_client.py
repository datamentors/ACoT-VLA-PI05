#!/usr/bin/env python3
"""Launch a Genie Sim benchmark task with the LeRobot-style RTC client (rtc_policy.py).

    python3 run_rtc.py [rtc flags] -- <app.py args>

No g2_vla_bridge, no chunk repairs. Everything after `--` goes to
geniesim_benchmark/app/app.py unchanged. Needs a server with method "rtc_infer"
(ACoT-VLA scripts/serve_policy.py, 2026-10-01 or later).

    # clean RTC, LeRobot defaults (H=10, w=10, threshold=30, 30 Hz)
    python3 run_rtc.py --trace /workspace/runs/rtc.jsonl -- --config ... --benchmark.infer_host=HOST:8000

    # ablation: identical async loop, guidance off (no leftover sent)
    python3 run_rtc.py --no-guidance --trace /workspace/runs/async.jsonl -- ...
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _split_argv(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-guidance", action="store_true", help="send no leftover: async chunking without RTC")
    p.add_argument("--execution-horizon", type=int, default=10)
    p.add_argument("--max-guidance-weight", type=float, default=10.0)
    p.add_argument("--queue-threshold", type=int, default=30, help="re-infer when queued rows <= this")
    p.add_argument("--pace-hz", type=float, default=30.0, help="wall-clock act() rate; 0 = unpaced")
    p.add_argument("--trace", default="")
    p.add_argument("--log-level", default="INFO")
    return p


def main():
    mine, theirs = _split_argv(sys.argv[1:])
    a = build_parser().parse_args(mine)

    import logging
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import rtc_policy
    rtc_policy.CFG = rtc_policy.RtcConfig(
        enabled=not a.no_guidance,
        execution_horizon=a.execution_horizon,
        max_guidance_weight=a.max_guidance_weight,
        queue_threshold=a.queue_threshold,
        pace_hz=a.pace_hz,
        trace=a.trace,
    )

    # corobotpolicy imports Isaac/omni modules, so it cannot be imported before
    # SimulationApp boots: swap CoRobotPolicy -> RtcPolicy when the benchmark imports it.
    import importlib.abc

    TARGET = "geniesim_benchmark.benchmark.policy.corobotpolicy"

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != TARGET:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                module.CoRobotPolicy = rtc_policy.build(module.CoRobotPolicy)
                print("[run_rtc] CoRobotPolicy -> RtcPolicy", flush=True)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())

    import geniesim_benchmark  # app.py boots SimulationApp at import: locate it, don't import it
    app_py = os.path.join(os.path.dirname(geniesim_benchmark.__file__), "app", "app.py")
    print(f"[run_rtc] rtc cfg: {rtc_policy.CFG}", flush=True)
    print(f"[run_rtc] launching {app_py} {' '.join(theirs)}", flush=True)
    sys.argv = [app_py] + theirs
    runpy.run_path(app_py, run_name="__main__")


if __name__ == "__main__":
    main()
