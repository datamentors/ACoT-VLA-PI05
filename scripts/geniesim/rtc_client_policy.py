"""LeRobot-style Real-Time Chunking client for Genie Sim. No g2_vla_bridge.

Port of LeRobot's RTC rollout (lerobot/rollout/inference/rtc.py RTCInferenceEngine +
lerobot/policies/rtc/action_queue.py ActionQueue) onto the stock CoRobotPolicy:

    sim thread (act, 1 row / policy step)        RTC thread
    ------------------------------------        ----------
    publish latest obs                  ----->  qsize <= queue_threshold ?
    pop next row from ActionQueue                 leftover = queue.get_left_over()   (absolute rows)
    hold last command if queue empty              delay    = max recent consumed-rows-per-inference
                                                  rtc_infer(obs, leftover, delay, H)
                                                  merge: queue = chunk[rows consumed meanwhile:]

What is deliberately NOT here (the old bridge's repairs): no max_rows tail chop, no
latency_skip, no replan_rows, no savgol / smoothing, no seam blend, no gripper skip,
no derivative gate. Full 30-row chunks, one row per sim policy step, exactly what the
server returns -> CoRobotPolicy._post_process_action (identical to the stock client).

Server side (ACoT-VLA openpi, method "rtc_infer"): re-anchors the absolute leftover to
the current state through the policy's own input transforms, then LeRobot guidance.

Time base: LeRobot measures delay as ceil(latency / dt). Here one queue row is one sim
policy step, so delay is counted directly as rows actually consumed during inference
(LeRobot's merge already uses min(latency_delay, rows_consumed)). act() is paced to
`pace_hz` wall clock (30 = the data / policy-step rate) so latency in rows matches a
real robot running at 30 Hz.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import threading
import time

import msgpack
import numpy as np

log = logging.getLogger("rtc_policy")


@dataclasses.dataclass
class RtcConfig:
    enabled: bool = True              # False = same async loop, no leftover sent (guidance off)
    execution_horizon: int = 10       # LeRobot RTCConfig default
    max_guidance_weight: float = 10.0  # LeRobot RTCConfig default
    queue_threshold: int = 30         # LeRobot rtc_queue_threshold default: re-infer when qsize <= this
    pace_hz: float = 30.0             # wall-clock rate of act(); 0 = run as fast as the sim goes
    trace: str = ""                   # jsonl, one line per inference (all episodes, "episode" field)
    prefix_attention_schedule: str | None = None  # "exp"/"linear"; None = server default (linear)


CFG = RtcConfig()


class ActionQueue:
    """LeRobot ActionQueue (RTC-enabled mode), numpy.

    The server re-anchors absolute leftovers itself, so the "original" and "processed"
    queues of LeRobot collapse into one queue of the server's raw rows; per-step
    post-processing (gripper mapping) runs at pop time, like the stock client.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.queue: np.ndarray | None = None
        self.last_index = 0

    def get(self) -> np.ndarray | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            row = self.queue[self.last_index].copy()
            self.last_index += 1
            return row

    def qsize(self) -> int:
        with self.lock:
            return 0 if self.queue is None else len(self.queue) - self.last_index

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index:].copy()

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.last_index = 0

    def merge(self, actions: np.ndarray, real_delay: int, action_index_before_inference: int | None) -> int:
        """Replace the queue with actions[delay:]; never skip more than was consumed."""
        with self.lock:
            delay = max(0, real_delay)
            if action_index_before_inference is not None:
                delay = min(delay, max(0, self.last_index - action_index_before_inference))
            delay = min(delay, len(actions))
            self.queue = np.asarray(actions[delay:], dtype=np.float64).copy()
            self.last_index = 0
            return delay


def _result_rows(inner: dict) -> np.ndarray:
    """Server result envelope -> [T, 16 or 21] rows in the server's own action layout."""
    cols = [
        np.asarray(inner["left_arm"]["values"], dtype=np.float64),
        np.asarray(inner["right_arm"]["values"], dtype=np.float64),
        np.asarray(inner["left_effector"], dtype=np.float64).reshape(-1, 1),
        np.asarray(inner["right_effector"], dtype=np.float64).reshape(-1, 1),
    ]
    waist = inner.get("waist")
    if isinstance(waist, dict) and waist.get("values") is not None:
        cols.append(np.asarray(waist["values"], dtype=np.float64))
    return np.concatenate(cols, axis=1)


def build(CoRobotPolicy):
    """Subclass the benchmark's CoRobotPolicy (importable only after SimulationApp boots)."""
    from geniesim_benchmark.utils.comm.retry import get_recv_timeout
    from geniesim_benchmark.utils.infer_post_process import get_arm_states

    class RtcPolicy(CoRobotPolicy):  # noqa: F821
        cfg: RtcConfig = CFG

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.queue = ActionQueue()
            self._obs_lock = threading.Lock()
            self._obs = None                  # (observation, instruction, gen_config)
            self._epoch = 0
            self._episode = 0
            self._step = 0
            self._last_out = None
            self._last_act_t = 0.0
            self._starved = 0
            self._delays = collections.deque(maxlen=100)   # LeRobot LatencyTracker, in rows
            self._n_infer = 0
            self._errors = 0
            self._failed = None
            self._stop = threading.Event()
            self._thread = None

        # -- benchmark hooks ------------------------------------------------------
        def need_infer(self):
            return True       # render every step: the RTC thread always needs a fresh image

        def inference_due(self):
            return False      # never settle-wait; the RTC thread decides when to infer

        def reset(self):
            super().reset()
            with self._obs_lock:
                self.queue.clear()
                self._obs = None
                self._epoch += 1           # in-flight chunk from the old episode is discarded
                self._last_out = None
                self._step = 0
                self._starved = 0
                self._episode += 1

        def update_task_status(self, done, task_progress):
            super().update_task_status(done, task_progress)
            if done:
                with self._obs_lock:
                    self.queue.clear()
                    self._epoch += 1

        def shutdown(self):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=3.0)
            super().shutdown()

        # -- control side (sim thread) -------------------------------------------
        def act(self, observation, **kwargs):
            if self._failed is not None:
                raise RuntimeError(f"RTC inference thread failed: {self._failed}")
            if self.cfg.pace_hz > 0:
                wait = self._last_act_t + 1.0 / self.cfg.pace_hz - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self._last_act_t = time.monotonic()

            instruction = kwargs.get("task_instruction", "")
            override = os.environ.get("GENIESIM_PROMPT_OVERRIDE")
            if override:
                instruction = override
            with self._obs_lock:
                self._obs = (observation, instruction, kwargs.get("gen_config"))
            if self._thread is None:
                self._thread = threading.Thread(target=self._rtc_loop, name="rtc-infer", daemon=True)
                self._thread.start()
            self._step += 1

            row = self.queue.get()
            if row is None:
                # LeRobot: empty queue -> no new action; the robot keeps its last target.
                self._starved += 1
                if self._last_out is not None:
                    return self._last_out
                cur = get_arm_states(observation["states"], self._arm_dim)
                return {"arm": [float(v) for v in cur[: self._arm_dim]]}

            entry = {
                "arm": row[0:14],
                "gripper": row[14:16],
                "kind": "JOINT_ABS",
                "eef_frame": "base_link",
            }
            if row.shape[0] >= 21:
                entry["waist"] = row[16:21]
            cur = get_arm_states(observation["states"], self._arm_dim)
            out = self._post_process_action(entry, cur, arm_base_tf=observation.get("arm_base_transform"))
            self._last_out = out
            return out

        # -- inference side (RTC thread) -----------------------------------------
        def _request(self, payload: dict) -> tuple[dict, dict]:
            self._ensure_connection()
            self._ws.send(msgpack.packb(payload))
            response = self._ws.recv(timeout=get_recv_timeout())
            if isinstance(response, str):
                raise RuntimeError(f"Server error: {response}")
            result = msgpack.unpackb(response, raw=False)
            if result.get("error"):
                raise RuntimeError(f"Server returned error: {result['error']}")
            return result["result"], result

        def _rtc_loop(self) -> None:
            while not self._stop.is_set():
                with self._obs_lock:
                    obs = self._obs
                    epoch_before = self._epoch
                if obs is None or self.queue.qsize() > self.cfg.queue_threshold:
                    time.sleep(0.005)
                    continue
                observation, instruction, gen_config = obs
                try:
                    t0 = time.monotonic()
                    idx_before = self.queue.get_action_index()
                    left_over = self.queue.get_left_over()
                    has_prev = left_over is not None and len(left_over) > 0
                    delay = int(max(self._delays)) if (has_prev and self._delays) else 0

                    payload = self.get_payload(observation, instruction, gen_config)
                    if payload is None:
                        time.sleep(0.005)
                        continue
                    payload["method"] = "rtc_infer"
                    payload["params"]["rtc"] = {
                        "prev_actions": left_over.tolist() if (has_prev and self.cfg.enabled) else None,
                        "inference_delay": delay,
                        "execution_horizon": int(self.cfg.execution_horizon),
                        "max_guidance_weight": float(self.cfg.max_guidance_weight),
                    }
                    if self.cfg.prefix_attention_schedule:
                        payload["params"]["rtc"]["prefix_attention_schedule"] = self.cfg.prefix_attention_schedule
                    t_sent = time.monotonic()
                    inner, raw = self._request(payload)
                    rows = _result_rows(inner)
                    self._send_depth = self._parse_need_depth(inner)
                    rtt = time.monotonic() - t_sent

                    consumed = self.queue.get_action_index() - idx_before
                    self._n_infer += 1
                    if self._n_infer > 1:                  # LeRobot drops warmup latency
                        self._delays.append(consumed)
                    self._errors = 0

                    # last row the robot executed from the old queue while this request was in flight
                    last_row = left_over[min(consumed, len(left_over)) - 1] if (has_prev and consumed > 0) else None
                    with self._obs_lock:
                        if epoch_before != self._epoch:
                            log.info("discarding RTC chunk computed before an episode reset")
                            continue
                        skipped = self.queue.merge(rows, consumed, idx_before)
                    self._write_trace({
                        "t": time.time(), "episode": self._episode, "step": self._step,
                        "infer": self._n_infer, "rtc": bool(self.cfg.enabled and has_prev),
                        "prev_rows": 0 if not has_prev else int(len(left_over)),
                        "delay_req": delay, "consumed": int(consumed), "skipped": int(skipped),
                        "H": int(self.cfg.execution_horizon), "rtt_ms": rtt * 1000.0,
                        "build_ms": (t_sent - t0) * 1000.0,
                        "server": raw.get("server_timing"), "server_rtc": raw.get("rtc"),
                        "chunk_rows": int(len(rows)), "qsize": self.queue.qsize(),
                        "starved_total": self._starved,
                        # seam = jump from the last row the robot executed to the first new row
                        "seam_arm_max": None if last_row is None or skipped >= len(rows)
                        else float(np.max(np.abs(rows[skipped, :14] - last_row[:14]))),
                        "first_row": rows[skipped].tolist() if skipped < len(rows) else None,
                    })
                except Exception as exc:  # noqa: BLE001
                    self._errors += 1
                    log.warning("RTC inference error %d/10: %s: %s", self._errors, type(exc).__name__, exc)
                    self._drop_connection()
                    if self._errors >= 10:
                        self._failed = f"{type(exc).__name__}: {exc}"
                        return
                    time.sleep(0.5)

        def _write_trace(self, rec: dict) -> None:
            if not self.cfg.trace:
                return
            try:
                with open(self.cfg.trace, "a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
            except OSError as exc:
                log.debug("trace write failed: %s", exc)

    RtcPolicy.cfg = CFG
    return RtcPolicy
