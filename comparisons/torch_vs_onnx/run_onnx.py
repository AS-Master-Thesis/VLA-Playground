"""
Run an ONNX walking policy on the 29-DoF G1 in MuJoCo (from rl_mjlab)
"""
import logging
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import onnxruntime as ort
from g1_walk_to_target.g1_env import G1WalkEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "../../external/unitree_rl_mjlab/g1/config/policy/velocity/v0/exported/policy.onnx"
CONFIG_PATH = "../../external/unitree_rl_mjlab/g1/config/policy/velocity/v0/params/deploy.yaml"
XML_PATH = "../../external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
UNITREE_MUJOCO_DIR = "../../external/unitree_mujoco/"


def create_onnx_session(model_path):
    providers = [
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL"
            },
        ),
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(model_path, providers=providers)
    logger.info("Active ONNX providers: %s", session.get_providers())
    for inp in session.get_inputs():
        print("Input:", inp.name, "Shape:", inp.shape, "Type:", inp.type)
    for out in session.get_outputs():
        print("Input:", out.name, "Shape:", out.shape, "Type:", out.type)

    return session


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    obs_config = config["observations"]
    gait_params = obs_config["gait_phase"].get("params", {})
    cmd_ranges = config["commands"]["base_velocity"]["ranges"]
    act_config = config["actions"]["JointPositionAction"]

    return {
        "joint_ids_map": np.asarray(config["joint_ids_map"], dtype=np.int64),
        "step_dt": float(config["step_dt"]),
        "stiffness": np.asarray(config["stiffness"], dtype=np.float32),
        "damping": np.asarray(config["damping"], dtype=np.float32),
        "default_joint_pos": np.asarray(config["default_joint_pos"], dtype=np.float32),
        "lin_vel_x_range": tuple(cmd_ranges["lin_vel_x"]),
        "lin_vel_y_range": tuple(cmd_ranges["lin_vel_y"]),
        "ang_vel_z_range": tuple(cmd_ranges["ang_vel_z"]),
        "heading_range": cmd_ranges.get("heading"),
        "action_clip": act_config.get("clip"),
        "action_scale": np.asarray(act_config["scale"], dtype=np.float32),
        "action_offset": np.asarray(act_config["offset"], dtype=np.float32),
        "base_ang_vel_scale": np.asarray(obs_config["base_ang_vel"]["scale"], dtype=np.float32),
        "projected_gravity_scale": np.asarray(obs_config["projected_gravity"]["scale"], dtype=np.float32),
        "velocity_commands_scale": np.asarray(obs_config["velocity_commands"]["scale"], dtype=np.float32),
        "gait_period": float(gait_params.get("period", 0.6)),
        "gait_phase_scale": np.asarray(obs_config["gait_phase"]["scale"], dtype=np.float32),
        "joint_pos_rel_scale": np.asarray(obs_config["joint_pos_rel"]["scale"], dtype=np.float32),
        "joint_vel_rel_scale": np.asarray(obs_config["joint_vel_rel"]["scale"], dtype=np.float32),
        "last_action_scale": np.asarray(obs_config["last_action"]["scale"], dtype=np.float32),
    }


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    config = load_config(CONFIG_PATH)

    env = G1WalkEnv(UNITREE_MUJOCO_DIR)
    model, data = env.model, env.data

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]
        print(f"jid={jid:2d} | qpos={qadr:2d} | qvel={vadr:2d} | name={name}")

    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        target_jid = model.actuator_trnid[aid, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_jid)
        print(f"act={aid:2d} | ctrl={aid:2d} | act_name={aname:24s} -> jid={target_jid:2d} ({jname})")

    num_joints = len(config["default_joint_pos"])
    assert model.nu == num_joints, (
        f"Actuator count mismatch: model.nu={model.nu}, deploy.yaml has {num_joints} joints. "
        f"Check XML_PATH ({XML_PATH}) and joint ordering."
    )

    sim_dt = float(model.opt.timestep)
    control_decimation = max(1, round(config["step_dt"] / sim_dt))
    policy_dt = control_decimation * sim_dt

    session = create_onnx_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_obs_dim = session.get_inputs()[0].shape[-1]

    data.qpos[7 : 7 + num_joints] = config["default_joint_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    last_action = np.zeros(num_joints, dtype=np.float32)
    target_dof_pos = config["default_joint_pos"].copy()
    zero_vel = np.zeros(num_joints, dtype=np.float32)

    obs_dim = 3 + 3 + 3 + 2 + num_joints + num_joints + num_joints  # 98
    if isinstance(expected_obs_dim, int):
        assert obs_dim == expected_obs_dim, f"Obs dim {obs_dim} != ONNX input {expected_obs_dim}"
    obs = np.zeros(obs_dim, dtype=np.float32)

    counter = 0
    logger.info("Starting 29-DoF ONNX walk loop (sim_dt=%.4f, decimation=%d, policy_dt=%.4f)...",
                sim_dt, control_decimation, policy_dt)

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            
            # 1. Low-level PD torque control at every physics sub-step
            q = data.qpos[7 : 7 + num_joints]
            dq = data.qvel[6 : 6 + num_joints]

            tau = pd_control(
                target_dof_pos,
                q,
                config["stiffness"],
                zero_vel,
                dq,
                config["damping"],
            )

            data.ctrl[:num_joints] = np.clip(
                tau,
                model.actuator_ctrlrange[:num_joints, 0],
                model.actuator_ctrlrange[:num_joints, 1],
            )

            mujoco.mj_step(model, data)

            counter += 1
            if counter % control_decimation == 0:
                sim_time = counter * sim_dt

                # Stand for first 2s, then command forward velocity within lin_vel_x_range
                if sim_time < 2.0:
                    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
                    
                # Refresh q/dq after latest physics substep
                q = data.qpos[7 : 7 + num_joints]
                dq = data.qvel[6 : 6 + num_joints]
                quat_wxyz = data.qpos[3:7]
                omega = data.qvel[3:6]
                grav = get_gravity_orientation(quat_wxyz)

                period = config["gait_period"]
                phase = (sim_time % period) / period
                gait_phase = np.array(
                    [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
                    dtype=np.float32,
                )

                obs_base_ang_vel = omega * config["base_ang_vel_scale"]
                obs_proj_grav = grav * config["projected_gravity_scale"]
                obs_cmd = cmd * config["velocity_commands_scale"]
                obs_gait = gait_phase * config["gait_phase_scale"]
                obs_q_rel = (q - config["default_joint_pos"]) * config["joint_pos_rel_scale"]
                obs_dq_rel = dq * config["joint_vel_rel_scale"]
                obs_last_act = last_action * config["last_action_scale"]

                obs = np.concatenate(
                    [
                        obs_base_ang_vel,
                        obs_proj_grav,
                        obs_cmd,
                        obs_gait,
                        obs_q_rel,
                        obs_dq_rel,
                        obs_last_act,
                    ],
                    dtype=np.float32,
                )

                raw_action = session.run([output_name], {input_name: obs[None, :]})[0][0]
                raw_action = np.asarray(raw_action, dtype=np.float32)
                if config["action_clip"] is not None:
                    low, high = config["action_clip"]
                    raw_action = np.clip(raw_action, low, high)

                last_action = raw_action.copy()
                target_dof_pos = config["action_offset"] + config["action_scale"] * raw_action
                
            viewer.sync()
            elapsed = time.perf_counter() - step_start
            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)


if __name__ == "__main__":
    main()