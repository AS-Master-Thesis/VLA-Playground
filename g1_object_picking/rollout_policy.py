"""
Closed-loop rollout of a trained LeRobot policy (SmolVLA) on the G1 in MuJoCo.

    On MacOS the viewer needs mjpython, e.g.
    mjpython rollout_policy.py --mujoco-repo ./unitree_mujoco \
        --checkpoint <checkpoint> \
        --color red --episodes 5
"""
import argparse
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
from lerobot.policies.utils import prepare_observation_for_inference

from g1_env import G1PickEnv, TABLE_TOP_Z, CUBE_HALF, CUBE_COLORS
from generate_dataset import sample_cubes

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def load_policy(checkpoint, device):
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post


def pick_image_key(policy):
    """The camera key the trained policy expects (e.g. observation.images.camera1)."""
    keys = [k for k in policy.config.image_features if "empty" not in k]
    return keys[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco-repo", default="./unitree_mujoco")
    ap.add_argument("--checkpoint", required=True, help=".../checkpoints/last/pretrained_model")
    ap.add_argument("--color", default="red", choices=list(CUBE_COLORS) + ["random"])
    ap.add_argument("--task", default=None, help="override the language instruction")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=80, help="policy steps per episode (10 Hz)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()

    device = torch.device(a.device)
    policy, pre, post = load_policy(a.checkpoint, device)
    img_key = pick_image_key(policy)
    print("policy image key:", img_key)

    env = G1PickEnv(a.mujoco_repo)
    rng = np.random.default_rng(a.seed)

    viewer = None
    if not a.headless:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [0.3, -0.1, 0.9]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 150, -25

    def sync():
        if viewer is not None:
            viewer.sync()
            time.sleep(0.002)          # ~ real time at 5 substeps per sync

    successes = 0
    for ep in range(a.episodes):
        color = rng.choice(list(CUBE_COLORS)) if a.color == "random" else a.color
        task = a.task or f"pick up the {color} cube"
        env.reset(sample_cubes(rng))
        policy.reset(); pre.reset(); post.reset()
        sync()
        print(f"\nepisode {ep + 1}/{a.episodes}: {task!r}")

        for t in range(a.max_steps):
            if viewer is not None and not viewer.is_running():
                return
            frame = {img_key: env.render(), "observation.state": env.state()}
            obs = prepare_observation_for_inference(frame, device, task, "unitree_g1_right_arm_gripper")
            with torch.inference_mode():
                action = policy.select_action(pre(obs))
                action = post(action)
            action = action.squeeze(0).cpu().numpy().astype(np.float64)
            env.step(action, on_substep=sync)

        lifted = env.cube_pos(color)[2] - (TABLE_TOP_Z + CUBE_HALF)
        ok = lifted > 0.05
        successes += int(ok)
        others = {c: round(float(np.linalg.norm(env.cube_pos(c)[:2])), 3) for c in CUBE_COLORS if c != color}
        print(f"  lifted {lifted * 100:.1f} cm -> {'SUCCESS' if ok else 'fail'}")

    print(f"\nsuccess: {successes}/{a.episodes}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()