import asyncio
import dataclasses
import http
import io
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from PIL import Image
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        rtc_state = _RtcState()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                if isinstance(obs, dict) and "method" in obs:
                    action = _handle_rpc(self._policy, obs, rtc_state)
                else:
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


@dataclasses.dataclass
class _RtcState:
    actions: np.ndarray | None = None
    row_index: int = 0


def _handle_rpc(policy: _base_policy.BasePolicy, request: dict, rtc_state: _RtcState) -> dict:
    method = request.get("method")
    params = request.get("params") or {}
    if method == "infer":
        actions = _infer_corobot_actions(policy, params)
        return {"result": _actions_to_corobot_result(actions)}
    if method == "rtc_start":
        actions = _infer_corobot_actions(policy, params)
        rtc_state.actions = actions
        rtc_state.row_index = 0
        state = _assemble_state(params)
        return {"result": _next_rtc_result(rtc_state, state)}
    if method == "rtc_next":
        state = np.asarray(params.get("state", []), dtype=np.float64)
        return {"result": _next_rtc_result(rtc_state, state)}
    if method == "rtc_reset":
        rtc_state.actions = None
        rtc_state.row_index = 0
        return {"result": {"reset": True, "needs_observation": True}}
    return {"error": f"unknown method: {method}"}


def _infer_corobot_actions(policy: _base_policy.BasePolicy, params: dict) -> np.ndarray:
    obs = {
        "images": _decode_images(params.get("images") or {}),
        "state": _assemble_state(params).astype(np.float32),
        "prompt": params.get("prompt", ""),
        "task_name": params.get("task_name"),
    }
    result = policy.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"policy returned actions with shape {actions.shape}, expected 2D")
    if not np.isfinite(actions[:, :16]).all():
        raise ValueError("policy returned non-finite arm/gripper actions")
    return actions


def _decode_images(images: dict) -> dict:
    key_map = {
        "head": "top_head",
        "top_head": "top_head",
        "hand_left": "hand_left",
        "hand_right": "hand_right",
    }
    decoded = {}
    for src, dst in key_map.items():
        if src not in images or dst in decoded:
            continue
        image = images[src]
        if isinstance(image, np.ndarray):
            decoded[dst] = image
            continue
        data = image.get("image_data") if isinstance(image, dict) else None
        if data is None:
            raise ValueError(f"image {src!r} has no image_data")
        decoded[dst] = np.asarray(Image.open(io.BytesIO(bytes(data))).convert("RGB"))
    missing = {"top_head", "hand_left", "hand_right"} - set(decoded)
    if missing:
        raise ValueError(f"missing images: {sorted(missing)}")
    return decoded


def _assemble_state(params: dict) -> np.ndarray:
    states = params.get("states") or {}
    arm = np.asarray(states.get("arm_joint_states", []), dtype=np.float64)
    waist = np.asarray(states.get("waist_joint_states", []), dtype=np.float64)
    gripper = np.asarray(states.get("gripper_states", []), dtype=np.float64)
    if arm.shape != (14,):
        raise ValueError(f"arm_joint_states shape {arm.shape}, expected (14,)")
    if waist.shape != (5,):
        raise ValueError(f"waist_joint_states shape {waist.shape}, expected (5,)")
    if gripper.shape != (2,):
        raise ValueError(f"gripper_states shape {gripper.shape}, expected (2,)")
    state = np.empty(21, dtype=np.float64)
    state[0:14] = arm
    state[14:16] = gripper
    state[16:21] = waist
    if not np.isfinite(state).all():
        raise ValueError("state has non-finite values")
    return state


def _actions_to_corobot_result(actions: np.ndarray) -> dict:
    result = {
        "left_arm": {"kind": "absolute_joint_position", "values": actions[:, 0:7].tolist()},
        "right_arm": {"kind": "absolute_joint_position", "values": actions[:, 7:14].tolist()},
        "left_effector": {"kind": "absolute_gripper_position", "values": actions[:, 14:15].tolist()},
        "right_effector": {"kind": "absolute_gripper_position", "values": actions[:, 15:16].tolist()},
    }
    if actions.shape[1] >= 21 and np.isfinite(actions[:, 16:21]).all():
        result["waist"] = {"kind": "absolute_joint_position", "values": actions[:, 16:21].tolist()}
    return result


def _next_rtc_result(rtc_state: _RtcState, current_state: np.ndarray) -> dict:
    actions = rtc_state.actions
    if actions is None or rtc_state.row_index >= len(actions):
        return {"needs_observation": True}
    row_index = rtc_state.row_index
    rtc_state.row_index += 1
    target = _target_21(actions[row_index], current_state)
    return {
        "target": {"kind": "absolute_joint_position", "values": target.tolist()},
        "row_index": row_index,
        "horizon": int(len(actions)),
        "needs_observation": False,
    }


def _target_21(action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    target = np.full(21, np.nan, dtype=np.float64)
    dims = min(action.shape[0], 21)
    target[:dims] = action[:dims]
    if target.shape[0] >= 21 and (not np.isfinite(target[16:21]).all()):
        if current_state.shape == (21,) and np.isfinite(current_state[16:21]).all():
            target[16:21] = current_state[16:21]
    if not np.isfinite(target[:16]).all():
        raise ValueError("target has non-finite arm/gripper values")
    return target
