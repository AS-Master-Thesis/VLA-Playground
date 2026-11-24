"""
G1 environment scene in MuJoCo with colored blocks.

Builds on Unitree's official model (unitree_mujoco/unitree_robots/g1/scene_29dof.xml) 
and adds three colored boxes/cubes via MjSpec.
"""
import os
import numpy as np
import mujoco

CUBE_HALF = 0.2
CUBE_COLORS = {
    "red": [0.9, 0.1, 0.1, 1.0],
    "green": [0.1, 0.75, 0.15, 1.0],
    "blue": [0.15, 0.25, 0.95, 1.0],
}
IMG_SIZE = 224


def _sample_box_xy(rng, placed, xy_range=(2.0, 5.0), min_sep=1.0, max_tries=200):
    low, high = xy_range
    for _ in range(max_tries):
        x = float(rng.uniform(low, high)) * float(rng.choice([-1.0, 1.0]))
        y = float(rng.uniform(low, high)) * float(rng.choice([-1.0, 1.0]))
        if all(np.hypot(x - px, y - py) >= min_sep for px, py in placed):
            placed.append((x, y))
            return x, y
    placed.append((x, y))
    return x, y


def build_spec(repo_dir):
    path = os.path.join(repo_dir, "unitree_robots", "g1", "scene_29dof.xml")
    spec = mujoco.MjSpec.from_file(path)

    rng = np.random.default_rng(seed=42)
    placed = []
    for name, rgba in CUBE_COLORS.items():
        x, y = _sample_box_xy(rng, placed)
        body = spec.worldbody.add_body(name=f"cube_{name}", pos=[x, y, CUBE_HALF])
        body.add_freejoint(name=f"cube_{name}_free")
        body.add_geom(
            name=f"cube_{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF] * 3,
            rgba=rgba,
            mass=0.5,
            contype=1,
            conaffinity=1,
            friction=[1.5, 0.01, 0.001],
        )

    torso = spec.body("torso_link")
    tilt = np.deg2rad(25)
    y_axis = [np.sin(tilt), 0.0, np.cos(tilt)]
    torso.add_camera(
        name="head",
        pos=[0.08, 0.0, 0.43],
        fovy=80,
        xyaxes=[0.0, -1.0, 0.0, *y_axis],
    )

    spec.option.timestep = 0.002
    return spec


class G1WalkEnv:
    """Minimal G1 floor environment with three colored blocks and a head camera."""

    def __init__(self, repo_dir, img_size=IMG_SIZE, fps=10, seed=42):
        self.model = build_spec(repo_dir).compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, img_size, img_size)
        self.fps = fps
        self.n_sub = max(1, int(round(1.0 / fps / self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed=seed)

        self.cube_qadr = {
            c: self.model.jnt_qposadr[self.model.joint(f"cube_{c}_free").id]
            for c in CUBE_COLORS
        }
        self.cube_body = {
            c: self.model.body(f"cube_{c}").id
            for c in CUBE_COLORS
        }
        mujoco.mj_forward(self.model, self.data)

    def cube_pos(self, color):
        return self.data.xpos[self.cube_body[color]].copy()

    def reset(self, cube_xy=None, settle_steps=200):
        mujoco.mj_resetData(self.model, self.data)
        placed = []
        for color in CUBE_COLORS:
            if cube_xy is not None and color in cube_xy:
                x, y = cube_xy[color]
            else:
                x, y = _sample_box_xy(self.rng, placed)
            qadr = self.cube_qadr[color]
            self.data.qpos[qadr:qadr + 7] = [x, y, CUBE_HALF, 1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        self.renderer.update_scene(self.data, camera="head")
        return self.renderer.render().copy()

    def step(self, ctrl=None, on_substep=None, every=5):
        if ctrl is not None:
            ctrl_arr = np.asarray(ctrl, dtype=np.float64)
            if ctrl_arr.shape != (self.model.nu,):
                raise ValueError(f"Expected ctrl shape ({self.model.nu},), got {ctrl_arr.shape}")
            self.data.ctrl[:] = ctrl_arr

        every = max(1, int(every))
        for i in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            if on_substep is not None and (i + 1) % every == 0:
                on_substep()