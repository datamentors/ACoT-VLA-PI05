"""Live check of the LeRobot-style rtc_infer endpoint (manipulation_pi05, sorting_packages)."""
import sys, time, json, cv2, msgpack, numpy as np
import websockets.sync.client as wsc

URI = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000"
rng = np.random.default_rng(0)
def jpg(h, w):
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img); return {"encoding": "JPEG", "image_data": buf.tobytes(), "height": h, "width": w}
IMGS = {"head": jpg(400, 640), "hand_left": jpg(480, 640), "hand_right": jpg(480, 640)}
ARM0 = [0.0, -0.6, 0.0, -1.2, 0.0, 0.3, 0.0, 0.0, -0.6, 0.0, -1.2, 0.0, 0.3, 0.0]
def params(arm):
    return {"images": IMGS, "states": {"arm_joint_states": list(arm), "gripper_states": [0.0, 0.0],
            "waist_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0], "head_joint_states": [0.0, 0.0, 0.0]},
            "prompt": "Grab the yellow package on the table with right arm.", "robot_type": "G2_omnipicker",
            "task_name": "sorting_packages", "episode_idx": 0}
ws = wsc.connect(URI, max_size=None, compression=None, open_timeout=30)
def call(method, p):
    t = time.monotonic(); ws.send(msgpack.packb({"method": method, "params": p}))
    r = msgpack.unpackb(ws.recv(timeout=600), raw=False); dt = (time.monotonic() - t) * 1000
    if r.get("error"): raise SystemExit(r["error"])
    res = r["result"]
    rows = np.concatenate([np.asarray(res["left_arm"]["values"]), np.asarray(res["right_arm"]["values"]),
                           np.asarray(res["left_effector"]).reshape(-1, 1), np.asarray(res["right_effector"]).reshape(-1, 1)]
                          + ([np.asarray(res["waist"]["values"])] if res.get("waist") else []), 1)
    return rows, dt, r.get("server_timing"), r.get("rtc")

for i in range(3): A, dt, st, _ = call("infer", params(ARM0))
print(f"plain infer   : rtt {dt:7.1f} ms server {st}")
for i in range(3): _, dt, st, info = call("rtc_infer", {**params(ARM0), "rtc": {"prev_actions": None}})
print(f"rtc no-prev   : rtt {dt:7.1f} ms server {st} rtc {info}")
P = {**params(ARM0), "rtc": {"prev_actions": A[5:].tolist(), "inference_delay": 4, "execution_horizon": 10, "max_guidance_weight": 10.0}}
for i in range(3): R, dt, st, info = call("rtc_infer", P)
print(f"rtc with prev : rtt {dt:7.1f} ms server {st} rtc {info}")
print("shape", A.shape, R.shape)

# Re-anchoring: shift the measured arm state by +0.15 rad on every arm joint, keep the
# leftover (absolute) fixed. Guided rows must move toward the ABSOLUTE leftover, i.e.
# closer to it than the unguided chunk at the shifted state is.
d = 0.15
arm1 = [v + d for v in ARM0]
B, *_ = call("infer", params(arm1))                       # unguided at shifted state
L = A[5:]                                                  # leftover rows (absolute)
G, *_ = call("rtc_infer", {**params(arm1), "rtc": {"prev_actions": L.tolist(), "inference_delay": 4,
              "execution_horizon": 10, "max_guidance_weight": 10.0}})
e_unguided = np.abs(B[:4, :14] - L[:4, :14]).mean()
e_guided = np.abs(G[:4, :14] - L[:4, :14]).mean()
e_wrong = np.abs(G[:4, :14] - (L[:4, :14] + d)).mean()   # where an un-re-anchored delta prefix would pull
print(f"rows 0-3 |chunk - leftover_abs|: unguided {e_unguided:.4f}  guided {e_guided:.4f}   guided vs leftover+shift {e_wrong:.4f}")
w = np.abs(G[10:, :14] - B[10:, :14]).mean()
print(f"rows 10+ |guided - unguided| (no guidance there, only coupling): {w:.4f}")
ok = e_guided < e_unguided and e_guided < e_wrong
print("REANCHOR_OK" if ok else "REANCHOR_FAIL")
