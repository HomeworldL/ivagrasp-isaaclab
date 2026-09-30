"""Display official and local dexterous hands in a persistent GUI simulation.

This follows the structure of IsaacLab's `scripts/demos/hands.py`, but extends
the hand set with locally converted assets from this repository.

Examples:

    TERM=xterm ./isaaclab.sh -p scripts/view_hand_usd.py
    TERM=xterm ./isaaclab.sh -p scripts/view_hand_usd.py --hands allegro shadow_hand
    TERM=xterm ./isaaclab.sh -p scripts/view_hand_usd.py --hands liberhand_right inspire_hand_right --oscillate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hands",
    nargs="+",
    default=[
        "allegro",
        "shadow_hand",
        "liberhand_right",
    ],
    choices=("allegro", "shadow_hand", "liberhand_right", "inspire_hand_right", "inspire_hand_left"),
    help="Hands to display. Defaults to all currently available hands in this repository.",
)
parser.add_argument("--oscillate", action="store_true", help="Apply oscillating joint demo.")
parser.add_argument(
    "--summary-json",
    type=Path,
    default=None,
    help="Optional JSON path to dump a summary for all displayed hands.",
)
parser.add_argument(
    "--print-actuator-debug",
    action="store_true",
    help="Print runtime PhysX actuator stiffness/damping and first-env target/position snapshots.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext


HAND_ORDER = ("allegro", "shadow_hand", "liberhand_right", "inspire_hand_right", "inspire_hand_left")


def _resolve_hand_profile(hand: str) -> dict:
    """Return a hand profile for scene construction."""
    if hand == "allegro":
        from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

        return {
            "name": hand,
            "hand_cfg": ALLEGRO_HAND_CFG,
            "usd_path": ALLEGRO_HAND_CFG.spawn.usd_path,
            "sim_timestep": 0.005,
        }
    if hand == "shadow_hand":
        from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

        return {
            "name": hand,
            "hand_cfg": SHADOW_HAND_CFG,
            "usd_path": SHADOW_HAND_CFG.spawn.usd_path,
            "sim_timestep": 0.005,
        }
    if hand == "liberhand_right":
        from dynamic_dexgrasp_lab.assets.hands.liberhand import (
            LIBERHAND_CFG,
            LIBERHAND_USD_PATH,
        )
        if not Path(LIBERHAND_USD_PATH).is_file():
            raise FileNotFoundError(f"Liberhand USD does not exist: {LIBERHAND_USD_PATH}")

        return {
            "name": hand,
            "hand_cfg": LIBERHAND_CFG,
            "usd_path": LIBERHAND_USD_PATH,
            "sim_timestep": 0.001,
        }
    if hand in ("inspire_hand_right", "inspire_hand_left"):
        from dynamic_dexgrasp_lab.assets.hands.inspire_hand import INSPIRE_HAND_LEFT_CFG, INSPIRE_HAND_RIGHT_CFG

        hand_cfg = INSPIRE_HAND_RIGHT_CFG if hand == "inspire_hand_right" else INSPIRE_HAND_LEFT_CFG
        if not Path(hand_cfg.spawn.usd_path).is_file():
            raise FileNotFoundError(f"Inspire Hand USD does not exist: {hand_cfg.spawn.usd_path}")
        return {
            "name": hand,
            "hand_cfg": hand_cfg,
            "usd_path": hand_cfg.spawn.usd_path,
            "sim_timestep": 0.001,
        }
    raise ValueError(f"Unknown hand: {hand}")


def _build_viewer_hand_cfg(profile: dict, prim_path: str) -> ArticulationCfg:
    hand_cfg = profile["hand_cfg"].replace(prim_path=prim_path)
    hand_cfg.spawn.activate_contact_sensors = False
    hand_cfg.spawn.rigid_props.disable_gravity = True
    hand_cfg.spawn.articulation_props.fix_root_link = True
    return hand_cfg


def _define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    env_origins = torch.zeros(num_origins, 3)
    num_cols = int(math.ceil(math.sqrt(num_origins)))
    num_rows = int(math.ceil(num_origins / num_cols))
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    return env_origins.tolist()


def _design_scene(selected_hands: list[str]) -> tuple[dict[str, Articulation], list[list[float]], list[dict]]:
    ordered_hands = [hand for hand in HAND_ORDER if hand in selected_hands]
    profiles = [_resolve_hand_profile(hand) for hand in ordered_hands]

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    origins = _define_origins(num_origins=len(profiles), spacing=0.8)
    scene_entities: dict[str, Articulation] = {}
    for index, profile in enumerate(profiles, start=1):
        origin_path = f"/World/Origin{index}"
        sim_utils.create_prim(origin_path, "Xform", translation=origins[index - 1])
        hand_cfg = _build_viewer_hand_cfg(profile, prim_path=f"{origin_path}/Robot")
        scene_entities[profile["name"]] = Articulation(hand_cfg)
    return scene_entities, origins, profiles


def _write_summary_json(summary_path: Path, profiles: list[dict], entities: dict[str, Articulation]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for profile in profiles:
        hand = entities[profile["name"]]
        payload.append(
            {
                "hand": profile["name"],
                "usd_path": profile["usd_path"],
                "num_joints": hand.num_joints,
                "joint_names": hand.joint_names,
                "num_bodies": hand.num_bodies,
                "body_names": hand.body_names,
                "num_fixed_tendons": hand.num_fixed_tendons,
                "num_spatial_tendons": hand.num_spatial_tendons,
            }
        )
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _print_scene_summary(profiles: list[dict], entities: dict[str, Articulation]) -> None:
    for profile in profiles:
        hand = entities[profile["name"]]
        print(f"hand         : {profile['name']}", flush=True)
        print(f"usd_path     : {profile['usd_path']}", flush=True)
        print(f"num_joints   : {hand.num_joints}", flush=True)
        print(f"num_bodies   : {hand.num_bodies}", flush=True)
        print(
            f"num_tendons  : {hand.num_fixed_tendons} fixed, {hand.num_spatial_tendons} spatial",
            flush=True,
        )


def _print_actuator_debug(profile_name: str, robot: Articulation, *, tag: str) -> None:
    stiff_row = None
    damp_row = None
    try:
        stiff = robot.root_physx_view.get_dof_stiffnesses()
        damp = robot.root_physx_view.get_dof_dampings()
        if stiff.ndim == 2:
            stiff_row = stiff[0]
            damp_row = damp[0]
        else:
            stiff_row = stiff
            damp_row = damp
        print(
            f"[ACT_DEBUG][{profile_name}][{tag}] stiffness(min/max)=({float(stiff.min()):.4f}, {float(stiff.max()):.4f}) "
            f"damping(min/max)=({float(damp.min()):.4f}, {float(damp.max()):.4f})",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - debug path
        print(f"[ACT_DEBUG][{profile_name}][{tag}] failed to query PhysX stiffness/damping: {exc}", flush=True)

    q = robot.data.joint_pos[0]
    q_target = robot.data.joint_pos_target[0]
    if stiff_row is None or damp_row is None:
        stiff_row = q.new_zeros(q.shape[0])
        damp_row = q.new_zeros(q.shape[0])
    print(f"[ACT_DEBUG][{profile_name}][{tag}] first-env joint snapshot:", flush=True)
    for i, name in enumerate(robot.joint_names):
        print(
            f"  {name:<20s} q={float(q[i]): .5f} target={float(q_target[i]): .5f}"
            f" stiff={float(stiff_row[i]): .5f} damp={float(damp_row[i]): .5f}",
            flush=True,
        )


def _run_simulator(
    sim: SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
    profiles: list[dict],
) -> None:
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    grasp_mode = 0
    home_pos = {name: robot.data.default_joint_pos.clone() for name, robot in entities.items()}
    home_vel = {name: robot.data.default_joint_vel.clone() for name, robot in entities.items()}
    while simulation_app.is_running():
        if count % 1000 == 0:
            sim_time = 0.0
            count = 0
            for index, profile in enumerate(profiles):
                robot = entities[profile["name"]]
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.write_joint_state_to_sim(home_pos[profile["name"]], home_vel[profile["name"]])
                robot.reset()
            print("[INFO]: Resetting hands state...", flush=True)

        if not args_cli.oscillate and count % 100 == 0:
            grasp_mode = 1 - grasp_mode

        for profile in profiles:
            robot = entities[profile["name"]]
            if args_cli.oscillate:
                joint_targets = home_pos[profile["name"]].clone()
                num_actuated = min(joint_targets.shape[1], 6)
                for joint_index in range(num_actuated):
                    amp = 0.2 + 0.3 * (joint_index % 3)
                    freq = 1.0 + 0.5 * joint_index
                    joint_targets[:, joint_index] = (
                        home_pos[profile["name"]][:, joint_index]
                        + amp * torch.sin(torch.tensor([freq * sim_time], device=sim.device))
                    )
            else:
                joint_targets = robot.data.soft_joint_pos_limits[..., grasp_mode]
            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1
        for index, profile in enumerate(profiles):
            robot = entities[profile["name"]]
            robot.update(sim_dt)


def main() -> None:
    try:
        sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
        sim = SimulationContext(sim_cfg)
        try:
            sim.set_camera_view(eye=[0.0, -2.2, 1.4], target=[0.0, 0.0, 0.3])
            scene_entities, scene_origins, profiles = _design_scene(args_cli.hands)
            scene_origins = torch.tensor(scene_origins, device=sim.device)
            sim.reset()
            for robot in scene_entities.values():
                robot.update(sim.cfg.dt)

            _print_scene_summary(profiles, scene_entities)
            if args_cli.print_actuator_debug:
                for profile in profiles:
                    robot = scene_entities[profile["name"]]
                    _print_actuator_debug(profile["name"], robot, tag="after_reset")
            if args_cli.summary_json is not None:
                _write_summary_json(args_cli.summary_json, profiles, scene_entities)

            print("[INFO]: Setup complete...", flush=True)
            _run_simulator(sim, scene_entities, scene_origins, profiles)
        finally:
            sim.clear_all_callbacks()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
