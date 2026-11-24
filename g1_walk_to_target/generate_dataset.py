"""
Dataset generator of a Unitree G1 walking towards a colored cube, made so that a vision-language-action
model (SmolVLA) can actually learn it (or at least try).

Run:
    python generate_dataset.py --policy-onnx <policy.onnx> --deploy-yaml <deploy.yaml> --dry-run --episodes 3

Info:
Adding upper body control later. For pick and place, the expert for the arms will need its own planner (IK or teleoperation)
"""
import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import onnxruntime as ort
import yaml
import mujoco
import mujoco.viewer

from g1_env import G1WalkEnv, CUBE_COLORS, IMG_SIZE

from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROMPTS = [
    "walk towards the {c} cube",
    "reach the {c} block",
    "stroll in direction of the {c} cube",
    "see that {c} block? walk towards it",
]

SETTLE_SECONDS = 1.5                # walking policy balances the robot with command 0 before we record
CAM_HALF_FOV = math.radians(35)     # camera is 80 deg wide (+-40), 5 deg safety margin
SEARCH_WZ = 0.5                     # turning speed (rad/s, left) when the target isn't visible
STATE_DIM = 2                       # dummy state, see note 6 above


# ---------------------------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------------------------
def wrap_to_pi(angle):
    """Angles are periodic (359 deg == -1 deg). This squeezes any angle into [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_wxyz(quat):
    """Heading of the robot (which way it faces, seen from above) from MuJoCo's quaternion."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_gravity_orientation(quat):
    """Where does 'down' point in the robot's own frame? Upright = (0, 0, -1). Used as balance sensor."""
    qw, qx, qy, qz = quat
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def has_fallen(data):
    pelvis_height = data.qpos[2]
    gravity_z = get_gravity_orientation(data.qpos[3:7])[2]  # -1 = upright, 0 = lying sideways
    return pelvis_height < 0.55 or gravity_z > -0.6


def target_visible(robot_xy, robot_yaw, target_xy):
    """Is the target inside the horizontal field of view of the head camera?"""
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    return abs(wrap_to_pi(math.atan2(dy, dx) - robot_yaw)) < CAM_HALF_FOV


# ---------------------------------------------------------------------------------------------
# scene layout
# ---------------------------------------------------------------------------------------------
def sample_cubes_in_front(rng, target_color, dist_range=(2.0, 5.0),
                          target_fov_deg=35.0, other_fov_deg=50.0, min_sep=1.8):
    """
    Cube positions in the robot's start frame (robot at the origin looking along +x).
    Target: within +-30 deg of straight ahead, so it's always in the camera image at the start.
    The two other cubes are distractors, also mostly in front, so the model has to really look at
    the COLOR (and the text prompt) instead of just walking to "the" cube.
    """
    while True:
        pts = {}
        for c in CUBE_COLORS:
            fov = target_fov_deg if c == target_color else other_fov_deg
            bearing = math.radians(rng.uniform(-fov, fov))
            r = rng.uniform(*dist_range)
            pts[c] = (r * math.cos(bearing), r * math.sin(bearing))
        xy = np.array(list(pts.values()))
        if all(np.linalg.norm(xy[i] - xy[j]) >= min_sep for i in range(3) for j in range(i + 1, 3)):
            return pts


# ---------------------------------------------------------------------------------------------
# the scripted "expert" (decides which velocity command to give the walking policy)
# ---------------------------------------------------------------------------------------------
def expert_command(robot_xy, robot_yaw, target_xy, stop_dist, max_vx=0.5, max_wz=0.8, min_vx=0.25):
    """
    Returns what the expert WANTS: [vx, vy, wz]  (vx forward m/s, wz turn rad/s, left = positive).
    Everything here must be explainable from the camera image, otherwise the VLA can't learn it.
    """
    dx = float(target_xy[0] - robot_xy[0])
    dy = float(target_xy[1] - robot_xy[1])
    dist = math.hypot(dx, dy)
    yaw_err = wrap_to_pi(math.atan2(dy, dx) - robot_yaw)   # how far off we are facing

    if dist <= stop_dist:                                  # arrived -> stand still
        return np.zeros(3, dtype=np.float32)

    if not target_visible(robot_xy, robot_yaw, target_xy):  # can't see it -> search (turn left)
        return np.array([0.0, 0.0, SEARCH_WZ], dtype=np.float32)

    wz = float(np.clip(2.0 * yaw_err, -max_wz, max_wz))
    if abs(yaw_err) > 0.30:                                # (~17 deg) first turn on the spot
        vx = 0.0
    else:
        # slow down when close, but never below min_vx: the walking policy treats commands
        # under 0.1 as "stand still", so a smaller speed would hover in front of the target.
        vx = float(np.clip(0.8 * (dist - stop_dist), min_vx, max_vx))
    return np.array([vx, 0.0, wz], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# one episode = one robot walking to one cube
# ---------------------------------------------------------------------------------------------
def run_episode(env, session, cfg, target_color, rng, max_sec=30.0, noise=True,
                viewer=None, realtime=False):
    """Returns (frames, success, info). frames = list of (image, state, action)."""
    env.reset(cube_xy=sample_cubes_in_front(rng, target_color), settle_steps=0)
    model, data = env.model, env.data
    n = len(cfg["default_joint_pos"])  # 29 joints

    # default standing pose. qpos[0:7] = pelvis pose, joints start at 7; qvel joints start at 6.
    default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    data.qpos[7:7 + n] = default_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    other_cubes_start = {c: env.cube_pos(c)[:2].copy() for c in CUBE_COLORS if c != target_color}
    stop_dist = 1.0

    kp = np.asarray(cfg["stiffness"], dtype=np.float64)
    kd = np.asarray(cfg["damping"], dtype=np.float64)
    act = cfg["actions"]["JointPositionAction"]
    act_scale = np.asarray(act["scale"], dtype=np.float64)
    act_offset = np.asarray(act["offset"], dtype=np.float64)
    obs_cfg = cfg["observations"]
    step_dt = float(cfg["step_dt"])                                  # 0.02 s -> policy at 50 Hz
    period = float(obs_cfg["gait_phase"]["params"]["period"])
    decimation = int(round(step_dt / model.opt.timestep))            # policy runs every 10th physics step
    tau_lo, tau_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    last_action = np.zeros(n, dtype=np.float32)
    target_q = default_q.copy()
    current_cmd = np.zeros(3, dtype=np.float32)   # command actually given to the walking policy
    global_phase = 0.0
    policy_substeps = 0

    def build_obs(cmd):
        """The 98 numbers the walking policy expects (order as in the deploy yaml)."""
        q = data.qpos[7:7 + n].astype(np.float32)
        dq = data.qvel[6:6 + n].astype(np.float32)
        ang_vel = data.qvel[3:6].astype(np.float32)
        gravity = get_gravity_orientation(data.qpos[3:7])
        # like Unitree's deploy code: gait clock is (0,0) when the command is tiny
        if np.linalg.norm(cmd) < 0.1:
            gait = np.zeros(2, dtype=np.float32)
        else:
            gait = np.array([math.sin(2 * math.pi * global_phase),
                             math.cos(2 * math.pi * global_phase)], dtype=np.float32)
        return np.concatenate([
            ang_vel * np.asarray(obs_cfg["base_ang_vel"]["scale"], dtype=np.float32),
            gravity * np.asarray(obs_cfg["projected_gravity"]["scale"], dtype=np.float32),
            cmd * np.asarray(obs_cfg["velocity_commands"]["scale"], dtype=np.float32),
            gait * np.asarray(obs_cfg["gait_phase"]["scale"], dtype=np.float32),
            (q - default_q.astype(np.float32)) * np.asarray(obs_cfg["joint_pos_rel"]["scale"], dtype=np.float32),
            dq * np.asarray(obs_cfg["joint_vel_rel"]["scale"], dtype=np.float32),
            last_action * np.asarray(obs_cfg["last_action"]["scale"], dtype=np.float32),
        ]).astype(np.float32)

    def apply_control():
        """Called after EVERY physics step (500 Hz): policy every 10th call, PD torque every call."""
        nonlocal last_action, target_q, global_phase, policy_substeps
        if policy_substeps % decimation == 0:
            global_phase = (global_phase + step_dt / period) % 1.0
            obs = build_obs(current_cmd)
            last_action = session.run([out_name], {in_name: obs[None, :]})[0][0].astype(np.float32)
            target_q = act_offset + act_scale * last_action.astype(np.float64)
        policy_substeps += 1
        tau = kp * (target_q - data.qpos[7:7 + n]) - kd * data.qvel[6:6 + n]
        data.ctrl[:] = np.clip(tau, tau_lo, tau_hi)
        if viewer is not None and round(data.time / model.opt.timestep) % 10 == 0:
            viewer.sync()
            if realtime:
                time.sleep(10 * model.opt.timestep)

    # ---- settle: walking policy runs with command 0 and balances the robot ----
    apply_control()
    for _ in range(int(round(SETTLE_SECONDS / (env.n_sub * model.opt.timestep)))):
        env.step(on_substep=apply_control, every=1)
        if has_fallen(data):
            return [], False, {"reason": "fell while settling"}

    frames = []
    n_frames = max(1, int(round(max_sec * env.fps)))
    needed_stop_frames = max(1, int(round(2.0 * env.fps)))
    stopped_frames, visible_frames = 0, 0
    max_step = np.array([0.05, 0.05, 0.1], dtype=np.float32)   # how much the command may change per frame
    disturbance = np.zeros(3, dtype=np.float32)

    for _ in range(n_frames):
        if has_fallen(data):
            return frames, False, {"reason": "fell", "frames": len(frames)}
        if viewer is not None and not viewer.is_running():
            return frames, False, {"reason": "viewer closed"}

        robot_xy = data.qpos[:2]
        robot_yaw = yaw_from_quat_wxyz(data.qpos[3:7])
        target_xy = env.cube_pos(target_color)[:2]
        dist = float(np.linalg.norm(target_xy - robot_xy))
        visible_frames += target_visible(robot_xy, robot_yaw, target_xy)

        # what the expert wants = the LABEL we save
        wanted = expert_command(robot_xy, robot_yaw, target_xy, stop_dist)

        # what we actually execute = wanted + a slowly wandering disturbance (only while moving).
        # this pushes the robot slightly off course, so the data also shows how to get back.
        executed = wanted.copy()
        if noise and np.linalg.norm(wanted) > 1e-4:
            disturbance = 0.9 * disturbance + rng.normal(0.0, [0.04, 0.0, 0.18]).astype(np.float32)
            executed = np.clip(wanted + disturbance, [0.0, 0.0, -1.0], [0.8, 0.0, 1.0])
        else:
            disturbance[:] = 0.0
        current_cmd = current_cmd + np.clip(executed - current_cmd, -max_step, max_step)

        # record BEFORE stepping: image, (dummy) state, label = clean expert command [vx, wz]
        frames.append((env.render(), np.zeros(STATE_DIM, dtype=np.float32),
                       np.array([wanted[0], wanted[2]], dtype=np.float32)))

        env.step(on_substep=apply_control, every=1)

        if np.linalg.norm(current_cmd) < 1e-4 and dist <= stop_dist + 0.15:
            stopped_frames += 1
            if stopped_frames >= needed_stop_frames:
                break
        else:
            stopped_frames = 0

    final_robot_xy = data.qpos[:2]
    final_dist = float(np.linalg.norm(env.cube_pos(target_color)[:2] - final_robot_xy))
    other_final_dists = [
        float(np.linalg.norm(env.cube_pos(c)[:2] - final_robot_xy))
        for c in CUBE_COLORS if c != target_color
    ]
    min_other_final_dist = min(other_final_dists, default=float("inf"))
    nearest_is_target_with_margin = final_dist < (min_other_final_dist - 0.5)

    others_moved = max((float(np.linalg.norm(env.cube_pos(c)[:2] - p))
                        for c, p in other_cubes_start.items()), default=0.0)
    ok = (not has_fallen(data) and stopped_frames >= needed_stop_frames
          and abs(final_dist - stop_dist) < 0.25 and others_moved < 0.05
          and nearest_is_target_with_margin)
    info = {
        "final_dist": round(final_dist, 3),
        "min_other_dist": round(min_other_final_dist, 3) if math.isfinite(min_other_final_dist) else None,
        "stop_dist": round(stop_dist, 3),
        "stopped_frames": stopped_frames,
        "others_moved": round(others_moved, 3),
        "target_visible": f"{visible_frames / max(1, len(frames)):.0%}",
    }
    return frames, ok, info


# ---------------------------------------------------------------------------------------------
# dataset writing (LeRobot v3 format)
# ---------------------------------------------------------------------------------------------
def make_lerobot_dataset(repo_id, root, fps, use_videos=True):
    features = {
        "observation.images.head": {
            "dtype": "video" if use_videos else "image",
            "shape": (IMG_SIZE, IMG_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": [f"dummy_{i}" for i in range(STATE_DIM)]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["cmd_vx", "cmd_wz"]},
    }
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root,
                                 robot_type="unitree_g1_29dof", use_videos=use_videos,
                                 image_writer_threads=4)


def save_preview(frames, path, every=20):
    Image.fromarray(np.concatenate([f[0] for f in frames[::every]], axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco-repo", default="../unitree_mujoco")
    ap.add_argument("--policy-onnx", required=True)
    ap.add_argument("--deploy-yaml", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--repo-id", default="local/g1_walk_to_target_v2")
    ap.add_argument("--root", default="./data/g1_walk_to_target_v2")
    ap.add_argument("--fps", type=int, default=10, help="must divide 50: 10, 25 or 50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--no-noise", action="store_true", help="turn off the command disturbance")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no LeRobot, only preview images")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="open MuJoCo window (use mjpython on mac)")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()

    with open(args.deploy_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(args.seed)
    env = G1WalkEnv(args.mujoco_repo, img_size=IMG_SIZE, fps=args.fps, seed=args.seed)
    session = ort.InferenceSession(args.policy_onnx, providers=["CPUExecutionProvider"])
    model = env.model
    n = len(cfg["default_joint_pos"])

    # sanity checks: they catch the silent bugs (wrong joint order, wrong timing, ...)
    decimation = int(round(float(cfg["step_dt"]) / model.opt.timestep))
    assert env.n_sub % decimation == 0, f"--fps {args.fps} must divide 50 (use 10, 25 or 50)"
    assert model.nu == n
    for i in range(n):
        assert model.jnt_qposadr[model.actuator_trnid[i, 0]] == 7 + i, f"motor {i} drives the wrong joint"
    assert session.get_inputs()[0].shape[-1] == 3 + 3 + 3 + 2 + 3 * n
    assert not np.all(model.actuator_ctrlrange == 0)
    print("sanity checks passed")

    colors = list(CUBE_COLORS.keys())
    viewer_cm = mujoco.viewer.launch_passive(model, env.data) if args.viewer else nullcontext(None)
    ds = None if args.dry_run else make_lerobot_dataset(args.repo_id, args.root, args.fps,
                                                        use_videos=not args.no_videos)
    os.makedirs("previews", exist_ok=True)

    saved, attempts, discarded, t0 = 0, 0, 0, time.time()
    with viewer_cm as viewer:
        while saved < args.episodes and attempts < args.episodes * 4:
            if viewer is not None and not viewer.is_running():
                break
            attempts += 1
            color = colors[saved % len(colors)]          # cycle colors -> balanced dataset
            frames, ok, info = run_episode(env, session, cfg, color, rng, max_sec=args.max_sec,
                                           noise=not args.no_noise, viewer=viewer,
                                           realtime=args.realtime)
            if not ok:
                discarded += 1
                print(f"  discarded attempt {attempts} ({color}): {info}")
                continue
            task = PROMPTS[int(rng.integers(len(PROMPTS)))].format(c=color)
            if saved < 3:
                save_preview(frames, f"previews/episode_{saved:03d}_{color}.png")
            if ds is not None:
                for img, state, action in frames:
                    ds.add_frame({"observation.images.head": img, "observation.state": state,
                                  "action": action, "task": task})
                ds.save_episode()
            saved += 1
            print(f"[{saved}/{args.episodes}] {task!r:45s} frames={len(frames)} {info} "
                  f"({time.time() - t0:.0f}s)")

    print(f"\nsaved {saved} episodes from {attempts} attempts ({discarded} discarded)")
    if ds is not None:
        ds.finalize()
        print(f"dataset written to {args.root}")
        if args.push:
            ds.push_to_hub(private=args.private, tags=["g1", "mujoco", "walk", "synthetic"])
            print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
