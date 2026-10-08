"""Genie Sim client for the server-owned RTC stream (scripts/serve_rtc_stream.py --rtc.clock external).

Counterpart of the harness rtc_policy.py with the RTC queue moved to the server,
the way anvil-embodied-ai runs it. Each act() is one control tick:

    act()                                      server (one connection)
    -----                                      ------
    obs + tick  ---------------------------->  store latest obs (worker refills the queue)
                                               pop one row, authorize its age
    JOINT_ABS row  <-------------------------  {"type": "cmd", "action": [21]}
      or hold the last command  <------------  {"type": "status", ...} (not ready / latched)

Everything else matches rtc_policy.py so the two can be compared: the same
CoRobotPolicy payload (images, states, prompt, robot_type), the same
_post_process_action, act() paced to 30 Hz wall clock, and the last command held
while no row is available. A new episode sends "reset" (readiness restarts); a
latch is followed by "rearm" once observations flow again, like Anvil's operator rearm.

Simulation time is not wall time (physics runs at ~0.33-0.5x), so the server must use
the external clock: it advances one row per act(), never by its own timer.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import time

import msgpack
import numpy as np

log = logging.getLogger("rtc_stream_policy")


@dataclasses.dataclass
class StreamConfig:
    pace_hz: float = 30.0  # wall-clock act() rate; 0 = run as fast as the sim goes
    trace: str = ""  # jsonl, one line per act()


CFG = StreamConfig()


def build(CoRobotPolicy):
    """Subclass the benchmark's CoRobotPolicy (importable only after SimulationApp boots)."""
    from geniesim_benchmark.utils.comm.retry import get_recv_timeout
    from geniesim_benchmark.utils.infer_post_process import get_arm_states

    class RtcStreamPolicy(CoRobotPolicy):  # noqa: F821
        cfg: StreamConfig = CFG

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._last_out = None
            self._last_act_t = 0.0
            self._reset_pending = True
            self._episode = 0
            self._step = 0
            self._counts = collections.Counter()

        # -- benchmark hooks ------------------------------------------------------
        def need_infer(self):
            return True  # render every step: the server needs a fresh observation each tick

        def inference_due(self):
            return False  # never settle-wait; the server decides when to infer

        def reset(self):
            super().reset()
            if self._step:
                log.info("episode %d done: %s", self._episode, dict(self._counts))
            self._reset_pending = True
            self._last_out = None
            self._episode += 1
            self._step = 0
            self._counts.clear()

        # -- one control tick -----------------------------------------------------
        def _call(self, message: dict) -> dict:
            self._ensure_connection()
            self._ws.send(msgpack.packb(message))
            reply = self._ws.recv(timeout=get_recv_timeout())
            if isinstance(reply, str):
                raise RuntimeError(f"server sent text: {reply}")
            reply = msgpack.unpackb(reply, raw=False)
            if reply.get("type") == "error":
                raise RuntimeError(f"server error: {reply.get('error')}")
            return reply

        def act(self, observation, **kwargs):
            if self.cfg.pace_hz > 0:
                wait = self._last_act_t + 1.0 / self.cfg.pace_hz - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self._last_act_t = time.monotonic()

            instruction = os.environ.get("GENIESIM_PROMPT_OVERRIDE") or kwargs.get("task_instruction", "")
            # Stock CoRobotPolicy keeps sending depth until a server reply says otherwise; the
            # stream server never uses it, and with depth every observation is ~6.6 MB.
            self._send_depth = False
            payload = self.get_payload(observation, instruction, kwargs.get("gen_config"))
            if payload is None:
                return None
            self._step += 1

            reply = None
            try:
                if self._reset_pending:
                    self._call({"type": "reset"})
                    self._reset_pending = False
                reply = self._call({"type": "obs", "params": payload["params"], "tick": True})
                if reply.get("type") == "status" and reply.get("state") == "latched":
                    self._counts["latched"] += 1
                    log.warning("server latched: %s; rearming", reply.get("reason"))
                    rearm = self._call({"type": "rearm"})
                    if not rearm.get("rearm_ok"):
                        log.warning("rearm refused: %s", rearm.get("rearm_reason"))
            except Exception as exc:  # noqa: BLE001
                # A new connection starts a fresh server session; reset on the next tick too.
                self._counts["errors"] += 1
                log.warning("RTC stream error: %s: %s", type(exc).__name__, exc)
                self._drop_connection()
                self._reset_pending = True
                reply = None

            cur = get_arm_states(observation["states"], self._arm_dim)
            if reply is not None and reply.get("type") == "cmd":
                row = np.asarray(reply["action"], dtype=np.float64)
                entry = {"arm": row[0:14], "gripper": row[14:16], "kind": "JOINT_ABS", "eef_frame": "base_link"}
                if row.shape[0] >= 21:
                    entry["waist"] = row[16:21]
                out = self._post_process_action(entry, cur, arm_base_tf=observation.get("arm_base_transform"))
                self._counts["cmd"] += 1
                self._write_trace(reply, row)
                self._last_out = out
                return out

            # No row this tick (readiness proof, latch, error): keep the last target.
            self._counts["hold"] += 1
            self._write_trace(reply, None)
            if self._last_out is not None:
                return self._last_out
            return {"arm": [float(v) for v in cur[: self._arm_dim]]}

        def _write_trace(self, reply: dict | None, row: np.ndarray | None) -> None:
            if not self.cfg.trace:
                return
            record = {
                "t": time.time(),
                "episode": self._episode,
                "step": self._step,
                "type": None if reply is None else reply.get("type"),
                "state": None if reply is None else reply.get("state"),
                "seq": None if reply is None else reply.get("seq"),
                "action_age_s": None if reply is None else reply.get("action_age_s"),
                "queue": None if reply is None else reply.get("queue"),
                "row": None if row is None else row.tolist(),
            }
            try:
                with open(self.cfg.trace, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                log.debug("trace write failed: %s", exc)

    RtcStreamPolicy.cfg = CFG
    return RtcStreamPolicy
