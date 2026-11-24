"""
Run a PyTorch walking policy on the 29-DoF G1 in MuJoCo (from rl_mjlab)
"""
import logging
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from g1_walk_to_target.g1_env import G1WalkEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "../../external/unitree_rl_gym/configs/g1.yaml"
UNITREE_MUJOCO_DIR = "./unitree_mujoco/"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    POLICY_PATH = config["policy_path"]
    XML_PATH = config["xml_path"]

    SIM_DT = config["simulation_dt"]
    CONTROL_DECIMATION = config["control_decimation"]
    NUM_ACTIONS = config["num_actions"]
    NUM_OBS = config["num_obs"]
    PERIOD = 0.8

    KPS = np.array(config["kps"], dtype=np.float32)
    KDS = np.array(config["kds"], dtype=np.float32)
    DEFAULT_ANGLES = np.array(config["default_angles"], dtype=np.float32)
    CMD_SCALE = np.array(config["cmd_scale"], dtype=np.float32)

    ANG_VEL_SCALE = config["ang_vel_scale"]
    DOF_POS_SCALE = config["dof_pos_scale"]
    DOF_VEL_SCALE = config["dof_vel_scale"]
    ACTION_SCALE = config["action_scale"]

model = mujoco.MjModel.from_xml_path(XML_PATH)

print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
print("\n--- Joints (skipping freejoint at index 0) ---")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    jtype = model.jnt_type[jid]
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    print(f"jid={jid:2d} | type={jtype} | name={name:28s} | qpos[{qadr:2d}] | qvel[{vadr:2d}]")

print("\n--- Actuators -> Target Joint ---")
for aid in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    # trnid[aid, 0] is typically the target joint id for joint actuators
    target_jid = model.actuator_trnid[aid, 0]
    jname = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_jid)
        if 0 <= target_jid < model.njnt
        else "N/A"
    )
    lo, hi = model.actuator_ctrlrange[aid]
    print(f"act={aid:2d} | name={aname:28s} -> joint={jname:28s} | range=[{lo:.1f}, {hi:.1f}]")
    

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


#========
# qpos starts at 7 + 12 = 19, qvel starts at 6 + 12 = 18
UPPER_DEFAULT = np.zeros(model.nu - NUM_ACTIONS, dtype=np.float32)
# Optional: slight natural elbow bend (left_elbow is act 18 -> idx 6, right_elbow is act 25 -> idx 13)
UPPER_DEFAULT[6] = 0.3   # left_elbow_joint
UPPER_DEFAULT[13] = 0.3  # right_elbow_joint

UPPER_KPS = np.array([
    100, 100, 100,                  # waist: yaw, roll, pitch
    40, 40, 40, 40, 10, 10, 10,     # left arm + wrist
    40, 40, 40, 40, 10, 10, 10,     # right arm + wrist
], dtype=np.float32)

UPPER_KDS = np.array([
    3, 3, 3,
    2, 2, 2, 2, 0.5, 0.5, 0.5,
    2, 2, 2, 2, 0.5, 0.5, 0.5,
], dtype=np.float32)
#========


def main():
    env = G1WalkEnv(UNITREE_MUJOCO_DIR)
    model, data = env.model, env.data
    model.opt.timestep = SIM_DT

    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()

    # Start in default leg stance
    data.qpos[7 : 7 + NUM_ACTIONS] = DEFAULT_ANGLES
    mujoco.mj_forward(model, data)

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    counter = 0

    logger.info("Starting motion.pt smoke test (stand 2s -> walk forward)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            tau_legs = pd_control(
                target_dof_pos,
                data.qpos[7 : 7 + NUM_ACTIONS],
                KPS,
                np.zeros_like(KDS),
                data.qvel[6 : 6 + NUM_ACTIONS],
                KDS,
            )
            NUM_UPPER = 17  # 3 waist + 14 left/right arm & wrist joints

            tau_upper = pd_control(
                UPPER_DEFAULT,
                data.qpos[7 + NUM_ACTIONS : 7 + NUM_ACTIONS + NUM_UPPER],
                UPPER_KPS,
                np.zeros_like(UPPER_KDS),
                data.qvel[6 + NUM_ACTIONS : 6 + NUM_ACTIONS + NUM_UPPER],
                UPPER_KDS,
            )

            data.ctrl[:NUM_ACTIONS] = tau_legs
            data.ctrl[NUM_ACTIONS:] = tau_upper
            mujoco.mj_step(model, data)

            counter += 1
            if counter % CONTROL_DECIMATION == 0:
                sim_time = counter * SIM_DT
                cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32) if sim_time < 2.0 else np.array([0.3, 0.0, 0.0], dtype=np.float32)

                qj = (data.qpos[7 : 7 + NUM_ACTIONS] - DEFAULT_ANGLES) * DOF_POS_SCALE
                dqj = data.qvel[6 : 6 + NUM_ACTIONS] * DOF_VEL_SCALE
                quat = data.qpos[3:7]
                omega = data.qvel[3:6] * ANG_VEL_SCALE
                grav = get_gravity_orientation(quat)

                phase = (sim_time % PERIOD) / PERIOD
                sin_phase = np.sin(2.0 * np.pi * phase)
                cos_phase = np.cos(2.0 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = grav
                obs[6:9] = cmd * CMD_SCALE
                obs[9 : 9 + NUM_ACTIONS] = qj
                obs[9 + NUM_ACTIONS : 9 + 2 * NUM_ACTIONS] = dqj
                obs[9 + 2 * NUM_ACTIONS : 9 + 3 * NUM_ACTIONS] = action
                obs[9 + 3 * NUM_ACTIONS : 9 + 3 * NUM_ACTIONS + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)

                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().cpu().numpy().squeeze()

                target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            viewer.sync()
            elapsed = time.perf_counter() - step_start
            if elapsed < SIM_DT:
                time.sleep(SIM_DT - elapsed)


if __name__ == "__main__":
    main()