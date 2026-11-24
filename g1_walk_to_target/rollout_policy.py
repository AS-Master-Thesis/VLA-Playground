"""
Closed-loop rollout of a trained LeRobot policy (SmolVLA) on the G1 in MuJoCo,
paired with an ONNX walking policy + PD controller for balance and locomotion.

On macOS the viewer needs mjpython, e.g.

    mjpython rollout_policy.py --mujoco-repo ../unitree_mujoco \
        --policy-onnx /Users/danielpappalardo/Desktop/Prep/rl_policies/unitree_rl_mjlab/g1/config/policy/velocity/v0/exported/policy.onnx \
        --deploy-yaml /Users/danielpappalardo/Desktop/Prep/rl_policies/unitree_rl_mjlab/g1/config/policy/velocity/v0/params/deploy.yaml \
        --checkpoint "../outputs/train/my_smolvla_walker_2026-11-24 14:59:19.630159/checkpoints/002500/pretrained_model" \
        --color red --episodes 5
"""
import argparse
import math
import time
import numpy as np
import onnxruntime as ort
import torch
import yaml
import mujoco

from g1_env import G1WalkEnv, CUBE_COLORS, IMG_SIZE
from generate_dataset_2 import sample_cubes_in_front, yaw_from_quat_wxyz, wrap_to_pi


SETTLE_SECONDS = 1.5


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def has_fallen(data):
    pelvis_height = data.qpos[2]
    gravity_z = get_gravity_orientation(data.qpos[3:7])[2]
    return pelvis_height < 0.55 or gravity_z > -0.6


def load_policy(checkpoint, device):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post


def pick_image_key(policy):
    """The camera key the trained policy expects (e.g. observation.images.head or camera1)."""
    keys = [k for k in policy.config.image_features if "empty" not in k]
    return keys[0] if keys else "observation.images.head"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco-repo", default="./unitree_mujoco")
    ap.add_argument("--policy-onnx", required=True)
    ap.add_argument("--deploy-yaml", required=True)
    ap.add_argument("--checkpoint", required=True, help=".../checkpoints/last/pretrained_model")
    ap.add_argument("--color", default="red", choices=list(CUBE_COLORS) + ["random"])
    ap.add_argument("--task", default=None, help="override the language instruction")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300, help="policy steps per episode (10 Hz)")
    ap.add_argument("--fps", type=int, default=10, help="must divide 50: 10, 25 or 50")
    ap.add_argument("--success-dist", type=float, default=1.05, help="max distance (m) to cube")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()

    from lerobot.policies.utils import prepare_observation_for_inference

    with open(a.deploy_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(a.device)
    policy, pre, post = load_policy(a.checkpoint, device)
    img_key = pick_image_key(policy)
    print("policy image key:", img_key)

    env = G1WalkEnv(a.mujoco_repo, img_size=IMG_SIZE, fps=a.fps, seed=a.seed)
    session = ort.InferenceSession(a.policy_onnx, providers=["CPUExecutionProvider"])
    model, data = env.model, env.data
    n = len(cfg["default_joint_pos"])

    decimation = int(round(float(cfg["step_dt"]) / model.opt.timestep))
    assert env.n_sub % decimation == 0, (
        f"--fps {a.fps} gives {env.n_sub} physics steps per frame, which isn't a multiple of {decimation}."
    )
    assert model.nu == n

    default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    kp = np.asarray(cfg["stiffness"], dtype=np.float64)
    kd = np.asarray(cfg["damping"], dtype=np.float64)
    act = cfg["actions"]["JointPositionAction"]
    act_scale = np.asarray(act["scale"], dtype=np.float64)
    act_offset = np.asarray(act["offset"], dtype=np.float64)
    obs_cfg = cfg["observations"]
    step_dt = float(cfg["step_dt"])
    period = float(obs_cfg["gait_phase"]["params"]["period"])
    tau_lo = model.actuator_ctrlrange[:, 0]
    tau_hi = model.actuator_ctrlrange[:, 1]

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    q_rel_start = 3 + 3 + 3 + 2  # 11
    dq_start = q_rel_start + n   # 40 for n=29

    rng = np.random.default_rng(a.seed)
    viewer = None
    if not a.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.3, -0.1, 0.9]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 150, -25

    last_action = np.zeros(n, dtype=np.float32)
    target_q = default_q.copy()
    current_cmd = np.zeros(3, dtype=np.float32)
    upper_override = None
    global_phase = 0.0
    policy_substeps = 0

    def build_walk_obs(cmd):
        q = data.qpos[7:7 + n].astype(np.float32)
        dq = data.qvel[6:6 + n].astype(np.float32)
        ang_vel = data.qvel[3:6].astype(np.float32)
        gravity = get_gravity_orientation(data.qpos[3:7])

        if np.linalg.norm(cmd) < 0.1:
            gait = np.zeros(2, dtype=np.float32)
        else:
            gait = np.array([
                math.sin(2 * math.pi * global_phase),
                math.cos(2 * math.pi * global_phase),
            ], dtype=np.float32)

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
        nonlocal last_action, target_q, global_phase, policy_substeps

        if policy_substeps % decimation == 0:
            global_phase = (global_phase + step_dt / period) % 1.0
            walk_obs = build_walk_obs(current_cmd)
            raw = session.run([out_name], {in_name: walk_obs[None, :]})[0][0].astype(np.float32)
            last_action = raw
            target_q = act_offset + act_scale * raw.astype(np.float64)

        policy_substeps += 1

        full_target = target_q.copy()

        q = data.qpos[7:7 + n]
        dq = data.qvel[6:6 + n]
        tau = kp * (full_target - q) - kd * dq
        # Pass 29 torques directly to data.ctrl (avoiding env.step's internal loop)
        data.ctrl[:] = np.clip(tau, tau_lo, tau_hi)

        if viewer is not None and round(data.time / model.opt.timestep) % 10 == 0:
            viewer.sync()
            if a.realtime:
                time.sleep(10 * model.opt.timestep)

    def step_physics_substeps(n_steps):
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            apply_control()

    successes = 0
    for ep in range(a.episodes):
        if viewer is not None and not viewer.is_running():
            break

        color = rng.choice(list(CUBE_COLORS)) if a.color == "random" else a.color
        task = a.task or f"walk towards the {color} cube"

        env.reset(cube_xy=sample_cubes_in_front(rng, color), settle_steps=0)
        data.qpos[7:7 + n] = default_q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        other_cubes_start = {
            c: env.cube_pos(c)[:2].copy() for c in CUBE_COLORS if c != color
        }

        last_action = np.zeros(n, dtype=np.float32)
        target_q = default_q.copy()
        current_cmd = np.zeros(3, dtype=np.float32)
        global_phase = 0.0
        policy_substeps = 0

        policy.reset()
        pre.reset()
        post.reset()

        # Settle with walking policy active at zero command
        apply_control()
        settle_substeps = int(round(SETTLE_SECONDS / model.opt.timestep))
        step_physics_substeps(settle_substeps)

        if has_fallen(data):
            print(f"episode {ep + 1}/{a.episodes} ({color}): FAILED (fell while settling)")
            continue

        target_start = env.cube_pos(color)[:2].copy()
        hold = 0
        ep_ok = False
        outcome = "MISS"

        for _ in range(a.max_steps):
            if viewer is not None and not viewer.is_running():
                return
            if has_fallen(data):
                outcome = "FELL"
                break

            frame = {img_key: env.render(), "observation.state": np.zeros(2, np.float32)}

            obs = prepare_observation_for_inference(frame, device, task, "unitree_g1_29dof")
            with torch.inference_mode():
                vla_action = policy.select_action(pre(obs))
                vla_action = post(vla_action)

            act_np = vla_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            max_step = np.array([0.1, 0.1, 0.2], dtype=np.float32)
            want_vx = float(np.clip(act_np[0], 0.0, 0.6))
            if want_vx < 0.15:
                want_vx = 0.0
            want = np.array([want_vx, 0.0, np.clip(act_np[1], -0.8, 0.8)], dtype=np.float32)
            current_cmd = current_cmd + np.clip(want - current_cmd, -max_step, max_step)

            step_physics_substeps(env.n_sub)

            if has_fallen(data):
                outcome = "FELL"
                break

            dists = {c: float(np.linalg.norm(env.cube_pos(c)[:2] - data.qpos[:2])) for c in CUBE_COLORS}
            nearest_cube = min(dists, key=lambda c: dists[c])
            yaw = yaw_from_quat_wxyz(data.qpos[3:7])
            tx, ty = env.cube_pos(color)[:2]
            facing = abs(wrap_to_pi(math.atan2(ty - data.qpos[1], tx - data.qpos[0]) - yaw)) < math.radians(25)
            at_target = (nearest_cube == color) and (dists[color] <= a.success_dist) and facing

            if at_target and np.linalg.norm(current_cmd) < 0.1:
                hold += 1
            else:
                hold = 0

            target_moved = float(np.linalg.norm(env.cube_pos(color)[:2] - target_start)) > 0.05
            others_moved = max(
                (float(np.linalg.norm(env.cube_pos(c)[:2] - p)) for c, p in other_cubes_start.items()),
                default=0.0,
            )
            kicked = target_moved or (others_moved >= 0.05)

            if hold >= 10 and not kicked:
                ep_ok = True
                outcome = "SUCCESS"
                break

            if kicked:
                outcome = "KICKED"
            elif nearest_cube != color and np.linalg.norm(current_cmd) < 0.1 and dists[nearest_cube] <= a.success_dist:
                outcome = "WRONG_CUBE"

        final_dist = float(np.linalg.norm(env.cube_pos(color)[:2] - data.qpos[:2]))
        if ep_ok:
            successes += 1
        print(
            f"episode {ep + 1}/{a.episodes} ({color}): {outcome} "
            f"(final_dist={final_dist:.3f}m, nearest={nearest_cube})"
        )

    print(f"\nsuccess: {successes}/{a.episodes}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()