"""
G1 tabletop pick environment in MuJoCo.

Builds on Unitree's official model (unitree_mujoco/unitree_robots/g1/scene_29dof.xml) 
and adds a table and three colored cubes via MjSpec.
"""
import os
import numpy as np
import mujoco

ARM_JOINTS = [f"right_{j}_joint" for j in
              ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_roll", "wrist_pitch", "wrist_yaw"]]
WRIST_BODY = "right_wrist_yaw_link"

CUBE_HALF = 0.02                    # 4 cm cubes
CUBE_COLORS = {"red": [0.9, 0.1, 0.1, 1], "green": [0.1, 0.75, 0.15, 1], "blue": [0.15, 0.25, 0.95, 1]}
TABLE_TOP_Z = 0.84
TABLE_CENTER = np.array([0.34, -0.10])     # x, y
TABLE_HALF = np.array([0.20, 0.35])        # x, y half-extents

FINGER_MAX = 0.03                   # max finger travel (m) -> gripper = 1.0 (closed)
TCP_X = 0.125                       # tool-center-point offset along wrist x (m)

IMG_SIZE = 224


def _set_damping(joint, value):
    """MuJoCo's Python spec API changed damping from scalar to array in newer versions."""
    try:
        joint.damping = value
    except TypeError:
        joint.damping = np.array([value, 0.0, 0.0])


def build_spec(repo_dir):
    path = os.path.join(repo_dir, "unitree_robots", "g1", "scene_29dof.xml")
    spec = mujoco.MjSpec.from_file(path)

    # --- fix the base, remove Unitree's torque motors -----------------------------------
    spec.delete(spec.joint("floating_base_joint"))
    for act in list(spec.actuators):
        spec.delete(act)

    # --- robot bodies: gravity compensation, no collisions ------------------------------
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
    for j in spec.joints:
        if j.name not in ARM_JOINTS:        # keep everything else rigid-ish
            _set_damping(j, 200.0)
            j.frictionloss = 5.0

    # --- lighting / floor already come from scene_29dof.xml; add a softer fill light -----

    # --- table --------------------------------------------------------------------------
    tb = spec.worldbody.add_body(name="table", pos=[TABLE_CENTER[0], TABLE_CENTER[1], 0])
    tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[TABLE_HALF[0], TABLE_HALF[1], 0.02], pos=[0, 0, TABLE_TOP_Z - 0.02],
                rgba=[0.55, 0.45, 0.35, 1], contype=1, conaffinity=1, friction=[1, 0.005, 0.0001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (TABLE_TOP_Z - 0.04) / 2],
                        pos=[sx * (TABLE_HALF[0] - 0.03), sy * (TABLE_HALF[1] - 0.03),
                             (TABLE_TOP_Z - 0.04) / 2],
                        rgba=[0.45, 0.35, 0.25, 1], contype=0, conaffinity=0)

    # --- cubes (free bodies) --------------------------------------------------------------
    for i, (name, rgba) in enumerate(CUBE_COLORS.items()):
        c = spec.worldbody.add_body(name=f"cube_{name}",
                                    pos=[0.3, -0.2 + 0.1 * i, TABLE_TOP_Z + CUBE_HALF])
        c.add_freejoint(name=f"cube_{name}_free")
        c.add_geom(name=f"cube_{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[CUBE_HALF] * 3, rgba=rgba, mass=0.03, contype=1, conaffinity=1,
                   friction=[1.5, 0.01, 0.001])

    # --- head camera (D435-like pose on the G1 head) -----------------------------------
    torso = spec.body("torso_link")
    # camera looks along its -z. xyaxes = (x_axis, y_axis) in torso frame.
    # forward (+x) and tilted ~40 deg down toward the table.
    tilt = np.deg2rad(58)
    y_axis = [np.sin(tilt), 0, np.cos(tilt)]
    torso.add_camera(name="head", pos=[0.08, 0.0, 0.43], fovy=80,
                     xyaxes=[0, -1, 0, *y_axis])

    # --- simple parallel gripper on the right wrist ----------------------------------------
    wrist = spec.body(WRIST_BODY)
    wrist.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.055, 0.02],
                   pos=[0.085, 0, 0], rgba=[0.25, 0.25, 0.25, 1], contype=0, conaffinity=0)
    wrist.add_site(name="tcp", pos=[TCP_X, 0, 0], size=[0.005] * 3, rgba=[1, 1, 0, 0.0])
    for side, name in ((+1, "gripper_l"), (-1, "gripper_r")):
        f = wrist.add_body(name=f"finger_{name}", pos=[0.125, side * 0.042, 0])
        fj = f.add_joint(name=f"{name}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE,
                         axis=[0, -side, 0], range=[0, FINGER_MAX], armature=0.001)
        _set_damping(fj, 2.0)
        f.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.04, 0.006, 0.008], rgba=[0.15, 0.15, 0.15, 1], mass=0.02,
                   contype=1, conaffinity=1, friction=[2.0, 0.02, 0.002])
        f.gravcomp = 1.0

    # --- position actuators: 7 arm joints + 2 fingers (fingers share one command) ------------
    arm_kp = [120, 120, 80, 80, 25, 25, 25]
    arm_kd = [8, 8, 6, 6, 2, 2, 2]
    for jn, kp, kd in zip(ARM_JOINTS, arm_kp, arm_kd):
        spec.add_actuator(name=jn.replace("_joint", "_pos"), target=jn,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          biasprm=[0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0])
    for name in ("gripper_l", "gripper_r"):
        spec.add_actuator(name=f"{name}_pos", target=f"{name}_joint",
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[300, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          biasprm=[0, -300, -10, 0, 0, 0, 0, 0, 0, 0],
                          ctrllimited=True, ctrlrange=[0, FINGER_MAX])

    spec.option.timestep = 0.002
    return spec


class G1PickEnv:
    """State = 7 right-arm joint angles + gripper closure in [0,1]. Action = same (absolute targets)."""

    def __init__(self, repo_dir, img_size=IMG_SIZE, fps=10):
        self.model = build_spec(repo_dir).compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, img_size, img_size)
        self.fps = fps
        self.n_sub = int(round(1.0 / fps / self.model.opt.timestep))
        m = self.model
        self.arm_jid = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jid])
        self.arm_vadr = np.array([m.jnt_dofadr[j] for j in self.arm_jid])
        self.arm_lo = m.jnt_range[self.arm_jid, 0]
        self.arm_hi = m.jnt_range[self.arm_jid, 1]
        self.fl_qadr = m.jnt_qposadr[m.joint("gripper_l_joint").id]
        self.fr_qadr = m.jnt_qposadr[m.joint("gripper_r_joint").id]
        self.act_arm = [m.actuator(n.replace("_joint", "_pos")).id for n in ARM_JOINTS]
        self.act_fl = m.actuator("gripper_l_pos").id
        self.act_fr = m.actuator("gripper_r_pos").id
        self.tcp = m.site("tcp").id
        self.cube_qadr = {c: m.jnt_qposadr[m.joint(f"cube_{c}_free").id] for c in CUBE_COLORS}
        self.cube_body = {c: m.body(f"cube_{c}").id for c in CUBE_COLORS}
        # slightly bent "ready" pose; solved properly in reset()
        # IK is restricted to a "natural" sub-range of the joint limits so it never returns
        # contorted elbow-up / twisted-wrist solutions.
        self.ik_lo = np.maximum(self.arm_lo, [-1.6, -1.2, -1.2, -1.05, -1.5, -1.5, -1.5])
        self.ik_hi = np.minimum(self.arm_hi, [0.5, 0.5, 1.2, 1.9, 1.5, 1.5, 1.5])
        self.q_nom = np.array([-0.3, -0.15, 0.0, -0.4, 0.0, 0.3, 0.0])
        self.home_q = self.q_nom.copy()
        self.home_q = np.clip(self.home_q, self.arm_lo, self.arm_hi)
        mujoco.mj_forward(self.model, self.data)
        self.home_q, err, tilt = self.solve_ik(HOME_TCP, GRASP_ROT, self.home_q, iters=400)
        assert err < 5e-3 and tilt < 5, f"home pose IK failed: {err:.4f} m, {tilt:.1f} deg"

    # ------------------------------------------------------------------ state helpers
    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    def gripper(self):
        return float(0.5 * (self.data.qpos[self.fl_qadr] + self.data.qpos[self.fr_qadr]) / FINGER_MAX)

    def state(self):
        return np.concatenate([self.arm_q(), [self.gripper()]]).astype(np.float32)

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def tcp_rot(self):
        return self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def cube_pos(self, color):
        return self.data.xpos[self.cube_body[color]].copy()

    # ------------------------------------------------------------------ reset / step
    def reset(self, cube_xy):
        """cube_xy: dict color -> (x, y) on the table."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = self.home_q
        for c, (x, y) in cube_xy.items():
            a = self.cube_qadr[c]
            self.data.qpos[a:a + 7] = [x, y, TABLE_TOP_Z + CUBE_HALF + 1e-3, 1, 0, 0, 0]
        self.data.ctrl[self.act_arm] = self.home_q
        self.data.ctrl[self.act_fl] = 0.0
        self.data.ctrl[self.act_fr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):                      # settle
            mujoco.mj_step(self.model, self.data)

    def step(self, action, on_substep=None, every=5):
        """action: 8-vector (7 arm joint targets [rad], gripper closure [0,1]). Advances 1/fps s.
        on_substep: optional callback fired every `every` physics steps (used to keep a viewer smooth)."""
        a = np.asarray(action, dtype=np.float64)
        self.data.ctrl[self.act_arm] = np.clip(a[:7], self.arm_lo, self.arm_hi)
        g = float(np.clip(a[7], 0, 1)) * FINGER_MAX
        self.data.ctrl[self.act_fl] = g
        self.data.ctrl[self.act_fr] = g
        for i in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            if on_substep is not None and (i + 1) % every == 0:
                on_substep()

    def render(self):
        self.renderer.update_scene(self.data, camera="head")
        return self.renderer.render().copy()

    # ------------------------------------------------------------------ IK (virtual, on a copy)
    def solve_ik(self, target_pos, target_rot, q_init, iters=200, pos_w=1.0, rot_w=0.3,
                 damping=1e-3, step_clip=0.15):
        """Damped least squares, 6-D (position + orientation), arm joints only.
        Works on a scratch MjData so the live simulation is untouched."""
        m = self.model
        if not hasattr(self, "_ik_data"):
            self._ik_data = mujoco.MjData(m)           # reused scratch data (much faster)
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        q = np.array(q_init, dtype=np.float64)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        prev = np.inf
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.tcp]
            R = d.site_xmat[self.tcp].reshape(3, 3)
            e_pos = target_pos - p
            # rotation error as axis-angle vector in world frame
            Re = target_rot @ R.T
            ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            if ang < 1e-6:
                e_rot = np.zeros(3)
            else:
                e_rot = ang / (2 * np.sin(ang)) * np.array([Re[2, 1] - Re[1, 2],
                                                            Re[0, 2] - Re[2, 0],
                                                            Re[1, 0] - Re[0, 1]])
            cur = np.linalg.norm(e_pos) + 0.3 * np.linalg.norm(e_rot)
            if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 0.02:
                break
            if abs(prev - cur) < 2e-6:                 # stagnated -> good enough
                break
            prev = cur
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.vstack([pos_w * jacp[:, self.arm_vadr], rot_w * jacr[:, self.arm_vadr]])
            e = np.concatenate([pos_w * e_pos, rot_w * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            dq = np.clip(dq, -step_clip, step_clip)
            # null-space pull toward a natural posture resolves the 7th (redundant) DoF
            Jp = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            dq += (np.eye(7) - Jp @ J) @ (0.3 * (self.q_nom - q))
            dq = np.clip(dq, -step_clip, step_clip)
            q = np.clip(q + dq, self.ik_lo, self.ik_hi)
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[self.tcp].copy()
        R = d.site_xmat[self.tcp].reshape(3, 3)
        pos_err = float(np.linalg.norm(target_pos - p))
        tilt = float(np.degrees(np.arccos(np.clip(R[:, 0] @ target_rot[:, 0], -1, 1))))
        return q, pos_err, tilt


def roty(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Gripper points forward (+x), pitched 10 deg downward, fingers open sideways (world y).
GRASP_ROT = roty(np.deg2rad(10))
HOME_TCP = np.array([0.30, -0.30, TABLE_TOP_Z + 0.14])