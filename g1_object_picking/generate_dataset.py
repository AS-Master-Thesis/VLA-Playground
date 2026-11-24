"""
Generates a LeRobot dataset of a Unitree G1 picking one of three colored cubes
in MuJoCo, based on a language instruction.

    python generate_dataset.py --mujoco-repo ../unitree_mujoco --episodes 100 \
        --repo-id YOUR_HF_USER/g1_pick_cubes --root ./data/g1_pick_cubes

Add --push to upload to the Hugging Face Hub when finished (run `hf auth login` first).
Add --dry-run to skip LeRobot and only write a preview montage (useful for debugging physics).

Per frame (10 Hz by default):
    observation.images.head : 224x224x3 head-camera RGB   (stored as mp4 by LeRobot)
    observation.state       : [7 right-arm joint angles (rad), gripper closure 0..1]
    action                  : [7 target joint angles (rad), gripper command 0..1]
    task                    : "pick up the red cube" (language instruction)
"""
import argparse
import os
import time
import numpy as np

from g1_env import (G1PickEnv, GRASP_ROT, TABLE_TOP_Z, CUBE_HALF, CUBE_COLORS, IMG_SIZE, ARM_JOINTS)

# Region of the table where the arm can comfortably reach (found via IK reachability sweep).
CUBE_X_RANGE = (0.26, 0.37)
CUBE_Y_RANGE = (-0.20, 0.04)
MIN_CUBE_DIST = 0.09            # fingers are ~9.4 cm wide when open -> keep neighbours apart

PROMPTS = [
    "pick up the {c} cube",
    "grab the {c} block",
    "lift the {c} cube off the table",
    "pick the {c} block up",
]

STATE_NAMES = [j.replace("_joint", "") for j in ARM_JOINTS] + ["right_gripper"]


# ----------------------------------------------------------------------------------------
# scene sampling
# ----------------------------------------------------------------------------------------
def sample_cubes(rng):
    """Random xy for each colored cube, with a minimum pairwise distance."""
    while True:
        pts = {c: (rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE)) for c in CUBE_COLORS}
        p = np.array(list(pts.values()))
        d = [np.linalg.norm(p[i] - p[j]) for i in range(3) for j in range(i + 1, 3)]
        if min(d) >= MIN_CUBE_DIST:
            return pts


# ----------------------------------------------------------------------------------------
# scripted expert
# ----------------------------------------------------------------------------------------
def min_jerk(s):
    """Smooth 0->1 profile, zero velocity & acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def build_plan(cube_xy, rng):
    """List of phases: (duration_s, tcp_start, tcp_end, gripper_start, gripper_end)."""
    gz = TABLE_TOP_Z + CUBE_HALF + 0.005            # tcp height for the grasp
    hover = gz + 0.10 + rng.uniform(-0.01, 0.01)
    lift = gz + 0.12
    x, y = cube_xy
    j = lambda t: t * rng.uniform(0.9, 1.1)         # a bit of timing variety
    from g1_env import HOME_TCP
    return [
        (j(1.5), HOME_TCP,                  np.array([x, y, hover]), 0.0, 0.0),   # move over cube
        (j(1.0), np.array([x, y, hover]),   np.array([x, y, gz]),    0.0, 0.0),   # descend
        (0.3,    np.array([x, y, gz]),      np.array([x, y, gz]),    0.0, 0.0),   # settle
        (j(0.8), np.array([x, y, gz]),      np.array([x, y, gz]),    0.0, 1.0),   # close
        (0.3,    np.array([x, y, gz]),      np.array([x, y, gz]),    1.0, 1.0),   # let grip build
        (j(1.2), np.array([x, y, gz]),      np.array([x, y, lift]),  1.0, 1.0),   # lift
        (0.5,    np.array([x, y, lift]),    np.array([x, y, lift]),  1.0, 1.0),   # hold
    ]


def run_episode(env, color, cube_xy, rng):
    """Roll out the scripted pick. Returns (frames, ok, info). frames: list of (img, state, action)."""
    env.reset(cube_xy)
    start = {c: env.cube_pos(c).copy() for c in CUBE_COLORS}
    q_prev = env.arm_q()
    frames = []
    max_ik_err = 0.0

    for dur, p0, p1, g0, g1 in build_plan(cube_xy[color], rng):
        n = max(1, int(round(dur * env.fps)))
        for k in range(n):
            s = min_jerk((k + 1) / n)
            tcp = p0 + (p1 - p0) * s
            grip = g0 + (g1 - g0) * s
            q, err, tilt = env.solve_ik(tcp, GRASP_ROT, q_prev, iters=80)
            max_ik_err = max(max_ik_err, err)
            action = np.concatenate([q, [grip]]).astype(np.float32)
            frames.append((env.render(), env.state(), action))
            env.step(action)
            q_prev = q

    # --- success check ----------------------------------------------------------------------
    cube_end = env.cube_pos(color)
    lifted = cube_end[2] - (TABLE_TOP_Z + CUBE_HALF)
    in_hand = np.linalg.norm(cube_end[:2] - env.tcp_pos()[:2])
    others_moved = max(np.linalg.norm(env.cube_pos(c) - start[c]) for c in CUBE_COLORS if c != color)
    ok = (lifted > 0.06) and (in_hand < 0.03) and (others_moved < 0.01) and (max_ik_err < 0.01)
    info = dict(lifted=float(lifted), in_hand=float(in_hand), others_moved=float(others_moved),
                max_ik_err=float(max_ik_err))
    return frames, ok, info


# ----------------------------------------------------------------------------------------
# LeRobot writer
# ----------------------------------------------------------------------------------------
def make_lerobot_dataset(repo_id, root, fps, use_videos=True):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.images.head": {
            "dtype": "video" if use_videos else "image",
            "shape": (IMG_SIZE, IMG_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=root,
        robot_type="unitree_g1_right_arm_gripper", use_videos=use_videos,
        image_writer_threads=4,
    )


def save_preview(frames, path, every=6):
    from PIL import Image
    imgs = [f[0] for f in frames[::every]]
    Image.fromarray(np.concatenate(imgs, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco-repo", default="../unitree_mujoco", help="path to cloned unitree_mujoco")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--repo-id", default="local/g1_pick_cubes")
    ap.add_argument("--root", default="./data/g1_pick_cubes")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true", help="push to the Hugging Face Hub when done")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no LeRobot, only preview images")
    ap.add_argument("--no-videos", action="store_true", help="store PNG images instead of mp4")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = G1PickEnv(args.mujoco_repo, fps=args.fps)
    colors = list(CUBE_COLORS)

    ds = None
    if not args.dry_run:
        ds = make_lerobot_dataset(args.repo_id, args.root, args.fps, use_videos=not args.no_videos)
    os.makedirs("previews", exist_ok=True)

    saved, attempts, t0 = 0, 0, time.time()
    fails = {"count": 0}
    while saved < args.episodes and attempts < args.episodes * 4:
        attempts += 1
        color = colors[saved % len(colors)]                 # balanced colors
        cube_xy = sample_cubes(rng)
        frames, ok, info = run_episode(env, color, cube_xy, rng)
        if not ok:
            fails["count"] += 1
            print(f"  discarded attempt {attempts}: {info}")
            continue
        task = PROMPTS[rng.integers(len(PROMPTS))].format(c=color)
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

    print(f"\nsaved {saved} episodes from {attempts} attempts ({fails['count']} discarded)")
    if ds is not None:
        ds.finalize()
        print(f"dataset written to {args.root}")
        if args.push:
            ds.push_to_hub(private=args.private, tags=["g1", "mujoco", "pick", "synthetic"])
            print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()