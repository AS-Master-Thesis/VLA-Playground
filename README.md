# Vision-Language-Action Models Playground

- **Repository Role:** `Playground, Tests and Experiments`
- **Author:** `Pappalardo Daniel`
- **Supervisors:** `Lee Dongheui, Yashuai Yan`
- **Institution:** `Technical University of Vienna`


## Thesis Project Ovierview

The aim of this thesis is to experiment with Vision-Language-Action (VLA) models on a Unitree G1 humanoid robot. More precisely, it explores combining a custom RL-based lower-body controller with a fine-tuned upper-body VLA to steer the humanoid's base and enable high-level reasoning, allowing the robot to navigate freely and execute text-prompted tasks—such as grabbing objects across a room—successfully.

### Project Structure (additional repos)
* **[`VLA Research`](https://github.com/AS-Master-Thesis/VLA-Research)** — Literature notes, reading summaries, tool and framework comparisons and early exploration.
* **[`VLA Playground`](https://github.com/AS-Master-Thesis/VLA-Playground)** — Quick prototypes, isolated scripts, proof-of-concepts, and "throwaway" tests before anything becomes clean code.
* **[`VLA Thesis`](https://github.com/AS-Master-Thesis/VLA-Thesis)** — Final thesis codebase, reproducible, polished source code with artifacts.


## What Lives Here

* `comparisons/` — Comparisons and benchmarks between different tools, frameworks, repos, policies etc.
* `g1_object_picking` — A MuJoCo dataset creation and LeRobot SmolVLA fine tuning for object picking
* `g1_walk_to_target` — A MuJoCo dataset creation and LeRobot SmolVLA fine tuning for spatial targeted walking


## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/AS-Master-Thesis/VLA-Playground.git
cd VLA-Playground

# 2. Create a venv and activate it (then install into that environment with uv)
python3 -m venv .venv
source .venv/bin/activate

# Install requirements.txt and vcstool into the active .venv using uv
uv pip install -r requirements.txt vcstool "setuptools<81"

# 3. Import sibling/external repositories defined in a .repos file
vcs import . < .repos