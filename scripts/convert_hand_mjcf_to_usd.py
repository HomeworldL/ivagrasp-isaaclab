"""Convert a hand MJCF asset to USD and export migration metadata.

Self-contained: all hand-specific parameters live in HAND_PROFILES below.
No imports from dynamic_dexgrasp_lab — the conversion must survive package
refactors and RL-hyperparameter changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Hand profiles — add new hands / left-right variants here.
# Every key is a valid --hand value.
#
# mimic_linear_specs  : (target_joint, source_joint, gearing, offset)
#   Derived from MJCF <equality><joint polycoef=...> by linearising around
#   the home pose.  For already-linear polycoef (c0 + c1*source):
#       gearing = -c1
#       offset  = -c0
#   All mimic joints are assumed to use the "rotZ" axis (revolute joints).
#
# mimic_natural_frequency / mimic_damping_ratio are hand-specific tuning
# parameters for the PhysX mimic coupling — they are NOT PhysX defaults.
# ---------------------------------------------------------------------------

HAND_PROFILES: dict[str, dict] = {
    # ========================================================================
    # Liberhand — right hand only for now
    #  20 joints, 13 actuated, 7 driven by 4th-order polynomial mimic couplings
    # ========================================================================
    "liberhand_right": {
        "canonical_xml": "source/dynamic_dexgrasp_lab/assets/hands/liberhand/xml/liberhand_right.xml",
        "articulation_root_suffix": "/hand_root/hand_root",
        "helper_body_names": (
            "hand_right_grasp",
            "hand_right_f14_end", "hand_right_f1_eef",
            "hand_right_f24_end", "hand_right_f2_eef",
            "hand_right_f34_end", "hand_right_f3_eef",
            "hand_right_f44_end", "hand_right_f4_eef",
            "hand_right_f54_end", "hand_right_f5_eef",
        ),
        "helper_container_names": ("hand_root",),
        "actuated_joint_names": (
            "hand_right_f11_joint", "hand_right_f12_joint", "hand_right_f13_joint",
            "hand_right_f21_joint", "hand_right_f22_joint", "hand_right_f23_joint",
            "hand_right_f31_joint", "hand_right_f32_joint",
            "hand_right_f41_joint", "hand_right_f42_joint",
            "hand_right_f51_joint", "hand_right_f52_joint", "hand_right_f53_joint",
        ),
        # Pre-linearised from 4th-order polycoef at home pose
        "mimic_linear_specs": (
            ("hand_right_f14_joint", "hand_right_f13_joint", -1.0, 0.0),
            ("hand_right_f24_joint", "hand_right_f23_joint", -1.0, 0.0),
            ("hand_right_f33_joint", "hand_right_f32_joint", -1.0, 0.0),
            ("hand_right_f34_joint", "hand_right_f33_joint", -1.0, 0.0),
            ("hand_right_f43_joint", "hand_right_f42_joint", -1.0, 0.0),
            ("hand_right_f44_joint", "hand_right_f43_joint", -1.0, 0.0),
            ("hand_right_f54_joint", "hand_right_f53_joint", -1.0, 0.0),
        ),
        "drive_stiffness": 3.0,
        "drive_damping": 0.1,
        "drive_max_force": 1.0,
        "mimic_natural_frequency": 0.0,
        "mimic_damping_ratio": 0.0,
        "contact_friction_static": 0.6,
        "contact_friction_dynamic": 0.6,
        "contact_restitution": 0.0,
    },
    # ========================================================================
    # Inspire Hand — right
    #  12 joints, 6 actuated, 6 driven by linear mimic couplings
    # ========================================================================
    "inspire_hand_right": {
        "canonical_xml": "source/dynamic_dexgrasp_lab/assets/hands/inspire_hand/xml/inspire_hand_right.xml",
        "articulation_root_suffix": "/hand_root/hand_root",
        "helper_body_names": (
            "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
        ),
        "helper_container_names": ("hand_root",),
        "actuated_joint_names": (
            "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
            "index_proximal_joint", "middle_proximal_joint",
            "ring_proximal_joint", "pinky_proximal_joint",
        ),
        # Linear MJCF equality polycoef = (c0, c1, 0, 0, 0)
        # PhysX: gearing = -c1, offset = -c0
        "mimic_linear_specs": (
            ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", -1.334, 0.0),
            ("thumb_distal_joint",      "thumb_proximal_pitch_joint", -0.667, 0.0),
            ("index_intermediate_joint", "index_proximal_joint", -1.06399, 0.04545),
            ("middle_intermediate_joint","middle_proximal_joint", -1.06399, 0.04545),
            ("ring_intermediate_joint",  "ring_proximal_joint",   -1.06399, 0.04545),
            ("pinky_intermediate_joint", "pinky_proximal_joint",  -1.06399, 0.04545),
        ),
        "drive_stiffness": 3.0,
        "drive_damping": 0.1,
        "drive_max_force": 1.0,
        "mimic_natural_frequency": 0.0,
        "mimic_damping_ratio": 0.0,
        "contact_friction_static": 0.6,
        "contact_friction_dynamic": 0.6,
        "contact_restitution": 0.0,
    },
    # ========================================================================
    # Inspire Hand — left  (same joint names as right)
    # ========================================================================
    "inspire_hand_left": {
        "canonical_xml": "source/dynamic_dexgrasp_lab/assets/hands/inspire_hand/xml/inspire_hand_left.xml",
        "articulation_root_suffix": "/hand_root/hand_root",
        "helper_body_names": (
            "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
        ),
        "helper_container_names": ("hand_root",),
        "actuated_joint_names": (
            "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
            "index_proximal_joint", "middle_proximal_joint",
            "ring_proximal_joint", "pinky_proximal_joint",
        ),
        "mimic_linear_specs": (
            ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", -1.334, 0.0),
            ("thumb_distal_joint",      "thumb_proximal_pitch_joint", -0.667, 0.0),
            ("index_intermediate_joint", "index_proximal_joint", -1.06399, 0.04545),
            ("middle_intermediate_joint","middle_proximal_joint", -1.06399, 0.04545),
            ("ring_intermediate_joint",  "ring_proximal_joint",   -1.06399, 0.04545),
            ("pinky_intermediate_joint", "pinky_proximal_joint",  -1.06399, 0.04545),
        ),
        "drive_stiffness": 3.0,
        "drive_damping": 0.1,
        "drive_max_force": 1.0,
        "mimic_natural_frequency": 400.0,
        "mimic_damping_ratio": 1.0,
        "contact_friction_static": 0.6,
        "contact_friction_dynamic": 0.6,
        "contact_restitution": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand", default="liberhand_right", choices=sorted(HAND_PROFILES.keys()),
        help="Hand asset profile to convert.",
    )
    parser.add_argument(
        "--xml", type=Path, default=None,
        help="Path to the source MJCF/XML. Overrides the profile canonical XML.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Defaults to <hand>/usd/<xml_stem>.",
    )
    parser.add_argument(
        "--fix-base", action="store_true",
        help="Import the hand as fixed-base (for debugging).",
    )
    parser.add_argument(
        "--make-instanceable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit instanceable USD output. Defaults to true.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    profile = HAND_PROFILES[args.hand]
    canonical = (REPO_ROOT / profile["canonical_xml"]).resolve()
    xml_path = Path(args.xml).resolve() if args.xml is not None else canonical
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = xml_path.parents[1] / "usd" / xml_path.stem
    return xml_path, output_dir, profile


# ---------------------------------------------------------------------------
# MJCF metadata parsing (generic — works for any hand)
# ---------------------------------------------------------------------------

def _parse_contact_excludes(root) -> list[dict[str, str]]:
    excludes: list[dict[str, str]] = []
    for exclude in root.findall("./contact/exclude"):
        excludes.append({
            "name": exclude.attrib.get("name", ""),
            "body1": exclude.attrib["body1"],
            "body2": exclude.attrib["body2"],
        })
    return excludes


def _parse_geom_rgba(root) -> dict[str, tuple[float, float, float, float]]:
    """Return {body_name: (r, g, b, a)} for every MJCF body with a geom."""
    body_rgba: dict[str, tuple[float, float, float, float]] = {}
    for body_elem in root.findall(".//body"):
        body_name = body_elem.attrib.get("name")
        if not body_name:
            continue
        for geom in body_elem.findall("geom"):
            if "rgba" in geom.attrib:
                rgba = tuple(float(v) for v in geom.attrib["rgba"].split())
                body_rgba[body_name] = rgba
                break
    return body_rgba


def _parse_equality_joints(root) -> list[dict]:
    records: list[dict] = []
    for joint in root.findall("./equality/joint"):
        polycoef = (
            [float(t) for t in joint.attrib.get("polycoef", "").split()]
            if joint.attrib.get("polycoef") else []
        )
        records.append({
            "joint1": joint.attrib["joint1"],
            "joint2": joint.attrib["joint2"],
            "polycoef": polycoef,
            "solref": joint.attrib.get("solref"),
            "solimp": joint.attrib.get("solimp"),
        })
    return records


def _parse_position_actuators(root) -> list[dict]:
    actuators: list[dict] = []
    for actuator in root.findall("./actuator/position"):
        actuators.append({
            "name": actuator.attrib.get("name", ""),
            "joint": actuator.attrib["joint"],
            "kp": float(actuator.attrib.get("kp", "0.0")),
            "forcerange": actuator.attrib.get("forcerange"),
            "ctrlrange": actuator.attrib.get("ctrlrange"),
        })
    return actuators


def _parse_mjcf_metadata(xml_path: Path) -> dict:
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    option = root.find("./option")
    compiler = root.find("./compiler")
    return {
        "mjcf_path": str(xml_path),
        "model_name": root.attrib.get("model"),
        "compiler": {
            "meshdir": compiler.attrib.get("meshdir") if compiler is not None else None,
            "autolimits": compiler.attrib.get("autolimits") if compiler is not None else None,
        },
        "option": {
            "gravity": option.attrib.get("gravity") if option is not None else None,
            "timestep": (
                float(option.attrib["timestep"])
                if option is not None and "timestep" in option.attrib else None
            ),
        },
        "contact_excludes": _parse_contact_excludes(root),
        "geom_rgba": {k: list(v) for k, v in _parse_geom_rgba(root).items()},
        "equality_joints": _parse_equality_joints(root),
        "position_actuators": _parse_position_actuators(root),
    }


# ---------------------------------------------------------------------------
# MJCF → USD conversion  (self-collision always enabled)
# ---------------------------------------------------------------------------


def _convert_mjcf(xml_path: Path, output_dir: Path, fix_base: bool, make_instanceable: bool) -> tuple[Path, object]:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    enable_extension("isaacsim.asset.importer.mjcf")

    class LocalMjcfConverter(MjcfConverter):
        def _get_mjcf_import_config(self):
            import_config = super()._get_mjcf_import_config()
            import_config.set_create_physics_scene(False)
            return import_config

    cfg = MjcfConverterCfg(
        asset_path=str(xml_path),
        usd_dir=str(output_dir),
        fix_base=fix_base,
        self_collision=True,
        make_instanceable=make_instanceable,
        import_inertia_tensor=True,
        force_usd_conversion=True,
    )
    converter = LocalMjcfConverter(cfg)
    return Path(converter.usd_path).resolve(), simulation_app


# ---------------------------------------------------------------------------
# Generic USD postprocess helpers
# ---------------------------------------------------------------------------

def _find_rigid_body_prims(stage):
    from pxr import UsdPhysics
    result = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result[prim.GetName()] = prim
    return result


def _find_joint_prims(stage):
    from pxr import UsdPhysics
    result = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            result[prim.GetName()] = prim
    return result


def _find_sublayer_path(usd_dir: Path, name_glob: str) -> Path | None:
    config_dir = usd_dir / "configuration"
    if not config_dir.is_dir():
        return None
    matches = sorted(config_dir.glob(name_glob))
    return matches[0] if matches else None


def _strip_mesh_prefix(mesh_name: str) -> str:
    for prefix in ("hand_right_", "hand_root_", "hand_base_", "right_", "left_"):
        if mesh_name.startswith(prefix):
            return mesh_name[len(prefix):]
    return mesh_name


# ---------------------------------------------------------------------------
# Postprocess steps (all generic)
# ---------------------------------------------------------------------------

def _collect_articulation_root_paths(stage) -> list[str]:
    from pxr import UsdPhysics

    return [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]


def _remove_articulation_apis(stage, prim_paths: list[str]) -> list[str]:
    from pxr import PhysxSchema, Sdf, UsdPhysics

    removed: list[str] = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if not prim.IsValid():
            continue
        removed_here = False
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed_here = True
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            removed_here = True
        if removed_here:
            removed.append(prim_path)
    return removed


def _apply_articulation_root_cleanup(root_usd: Path, physics_usd: Path | None, preferred_suffix: str | None) -> dict:
    """Keep only the preferred articulation root when possible."""
    from pxr import Usd

    if preferred_suffix is None:
        return {
            "articulation_root_paths_before": [],
            "kept_articulation_root_path": None,
            "extra_articulation_roots_before": [],
            "removed_articulation_roots": [],
            "articulation_root_paths_after": [],
            "remaining_extra_articulation_roots": [],
        }

    main_stage = Usd.Stage.Open(str(root_usd))
    articulation_paths_before = _collect_articulation_root_paths(main_stage)
    kept = next((path for path in articulation_paths_before if path.endswith(preferred_suffix)), None)
    extra = [path for path in articulation_paths_before if path != kept]

    if kept is None:
        raise RuntimeError(
            f"Failed to resolve preferred articulation root suffix {preferred_suffix!r} in {articulation_paths_before}."
        )

    removed_paths: list[str] = []
    if physics_usd is not None:
        physics_stage = Usd.Stage.Open(str(physics_usd))
        removed_paths.extend(_remove_articulation_apis(physics_stage, extra))
        physics_stage.GetRootLayer().Save()

    main_stage = Usd.Stage.Open(str(root_usd))
    articulation_paths_after = _collect_articulation_root_paths(main_stage)
    remaining_extra = [path for path in articulation_paths_after if path != kept]
    return {
        "articulation_root_paths_before": articulation_paths_before,
        "kept_articulation_root_path": kept,
        "extra_articulation_roots_before": extra,
        "removed_articulation_roots": sorted(set(removed_paths)),
        "articulation_root_paths_after": articulation_paths_after,
        "remaining_extra_articulation_roots": remaining_extra,
    }


def _apply_worldbody_cleanup(root_usd: Path, base_usd: Path | None, physics_usd: Path | None) -> dict:
    """Remove an empty importer-generated worldBody wrapper when safe."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    world_body_path = Sdf.Path("/liberhand_right/worldBody")
    stage = Usd.Stage.Open(str(root_usd))
    world_body = stage.GetPrimAtPath(world_body_path)
    if not world_body.IsValid():
        return {"found": False, "removed": False, "reason": "worldBody missing"}

    children = list(world_body.GetChildren())
    has_physics_api = any(
        world_body.HasAPI(api)
        for api in (
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            PhysxSchema.PhysxArticulationAPI,
            PhysxSchema.PhysxRigidBodyAPI,
        )
    )
    if children:
        return {
            "found": True,
            "removed": False,
            "reason": "worldBody has children",
            "child_paths": [str(child.GetPath()) for child in children],
        }
    if has_physics_api:
        return {
            "found": True,
            "removed": False,
            "reason": "worldBody still has physics APIs",
            "applied_schemas": list(world_body.GetAppliedSchemas()),
        }

    removed_in_layers: list[str] = []
    for sublayer_path in (base_usd, physics_usd):
        if sublayer_path is None:
            continue
        sub_stage = Usd.Stage.Open(str(sublayer_path))
        sub_world_body = sub_stage.GetPrimAtPath(world_body_path)
        if not sub_world_body.IsValid():
            continue
        sub_stage.RemovePrim(world_body_path)
        sub_stage.GetRootLayer().Save()
        removed_in_layers.append(sublayer_path.name)

    stage = Usd.Stage.Open(str(root_usd))
    return {
        "found": True,
        "removed": not stage.GetPrimAtPath(world_body_path).IsValid(),
        "reason": "removed empty worldBody" if removed_in_layers else "worldBody not authored in removable sublayers",
        "removed_in_layers": removed_in_layers,
    }


def _apply_helper_body_cleanup(stage, helper_body_names: tuple[str, ...]) -> list[dict[str, object]]:
    """Demote helper bodies to pure frame Xforms.

    MJCF importer currently creates helper frames such as ``*_end``, ``*_eef``,
    and ``grasp`` as rigid bodies plus a matching fixed joint under
    ``/liberhand_right/joints``. For runtime parity with official hand assets,
    these helper frames should remain addressable Xforms but must not stay in
    the physics chain.
    """
    from pxr import PhysxSchema, UsdPhysics

    helper_set = set(helper_body_names)
    cleaned: list[dict[str, object]] = []
    for prim in stage.Traverse():
        prim_name = prim.GetName()
        if prim_name not in helper_set:
            continue
        path_text = str(prim.GetPath())
        if path_text.startswith("/visuals/") or path_text.startswith("/collisions/") or path_text.startswith("/joints/"):
            continue

        removed_apis: list[str] = []
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI, UsdPhysics.FilteredPairsAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                removed_apis.append(api.__name__)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            removed_apis.append("PhysxRigidBodyAPI")

        removed_joint_paths: list[str] = []
        asset_root_path = prim.GetPath().GetParentPath().GetParentPath()
        joints_root = stage.GetPrimAtPath(asset_root_path.AppendChild("joints"))
        if joints_root.IsValid():
            for joint_prim in joints_root.GetChildren():
                body1_rel = joint_prim.GetRelationship("physics:body1")
                body1_targets = body1_rel.GetTargets() if body1_rel else []
                if prim.GetPath() in body1_targets or joint_prim.GetName() == prim_name:
                    removed_joint_paths.append(str(joint_prim.GetPath()))
                    stage.RemovePrim(joint_prim.GetPath())

        cleaned.append(
            {
                "helper_body_path": str(prim.GetPath()),
                "removed_apis": removed_apis,
                "removed_joint_paths": removed_joint_paths,
            }
        )
    return cleaned


def _apply_helper_descendant_cleanup(stage, helper_names: tuple[str, ...]) -> list[dict[str, object]]:
    """Remove helper-owned visuals/collisions subtrees while keeping the helper prim itself."""
    helper_set = set(helper_names)
    cleaned: list[dict[str, object]] = []
    for prim in stage.Traverse():
        if prim.GetName() not in helper_set:
            continue
        path_text = str(prim.GetPath())
        if path_text.startswith("/visuals/") or path_text.startswith("/collisions/") or path_text.startswith("/joints/"):
            continue
        removed_children: list[str] = []
        for child_name in ("visuals", "collisions"):
            child_path = prim.GetPath().AppendChild(child_name)
            child_prim = stage.GetPrimAtPath(child_path)
            if child_prim.IsValid():
                removed_children.append(str(child_path))
                stage.RemovePrim(child_path)
        if removed_children:
            cleaned.append(
                {
                    "helper_path": str(prim.GetPath()),
                    "removed_children": removed_children,
                }
            )
    return cleaned


def _apply_root_body_mass_fix(stage, articulation_root_suffix: str | None) -> dict[str, object]:
    """Patch importer-created dummy root rigid body with valid positive mass/inertia."""
    from pxr import Gf, UsdPhysics

    if articulation_root_suffix is None:
        return {"applied": False, "reason": "no articulation root suffix configured"}

    root_prim = None
    for prim in stage.Traverse():
        if str(prim.GetPath()).endswith(articulation_root_suffix):
            root_prim = prim
            break
    if root_prim is None or not root_prim.IsValid():
        return {"applied": False, "reason": "articulation root prim not found"}

    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return {"applied": False, "reason": "articulation root is not a rigid body"}

    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_value = 1.0e-4
    inertia_value = Gf.Vec3f(1.0e-8, 1.0e-8, 1.0e-8)
    mass_api.CreateMassAttr().Set(mass_value)
    mass_api.CreateDiagonalInertiaAttr().Set(inertia_value)
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    return {
        "applied": True,
        "root_body_path": str(root_prim.GetPath()),
        "mass": mass_value,
        "diagonal_inertia": [float(inertia_value[0]), float(inertia_value[1]), float(inertia_value[2])],
    }


def _apply_drive_settings(
    stage,
    actuated_joint_names: tuple[str, ...],
    stiffness: float, damping: float, max_force: float,
) -> list[str]:
    """Update existing drive parameters on actuated joints without changing drive type or targets."""
    from pxr import UsdPhysics
    joint_by_name = _find_joint_prims(stage)
    configured: list[str] = []
    for joint_name in actuated_joint_names:
        jp = joint_by_name.get(joint_name)
        if jp is None:
            raise KeyError(f"Actuated joint {joint_name!r} not found in stage.")
        drive_type = "angular" if jp.IsA(UsdPhysics.RevoluteJoint) else "linear"
        drive_api = UsdPhysics.DriveAPI.Get(jp, drive_type)
        if not drive_api:
            raise KeyError(f"DriveAPI for joint {joint_name!r} ({drive_type}) not found in stage.")
        drive_api.CreateStiffnessAttr().Set(stiffness)
        drive_api.CreateDampingAttr().Set(damping)
        drive_api.CreateMaxForceAttr().Set(max_force)
        configured.append(str(jp.GetPath()))
    return configured


def _apply_mimic_joints(
    stage,
    mimic_linear_specs: tuple[tuple[str, str, float, float], ...],
    natural_frequency: float, damping_ratio: float,
) -> list[dict]:
    """Rebuild mimic couplings as PhysX mimic metadata in *stage*."""
    from pxr import PhysxSchema, Sdf, UsdPhysics
    joint_by_name = _find_joint_prims(stage)
    configured: list[dict] = []
    for target_name, source_name, gearing, offset in mimic_linear_specs:
        target_jp = joint_by_name.get(target_name)
        source_jp = joint_by_name.get(source_name)
        if target_jp is None or source_jp is None:
            raise KeyError(f"Mimic pair {source_name!r} -> {target_name!r} not found in stage.")
        axis_token = "rotX"
        target_axis = getattr(UsdPhysics.Tokens, axis_token)
        reference_axis = getattr(UsdPhysics.Tokens, axis_token)
        mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(target_jp, target_axis)
        mimic_api.GetReferenceJointRel().ClearTargets(True)
        mimic_api.GetReferenceJointRel().AddTarget(source_jp.GetPath())
        mimic_api.GetReferenceJointAxisAttr().Set(reference_axis)
        mimic_api.GetGearingAttr().Set(float(gearing))
        mimic_api.GetOffsetAttr().Set(float(offset))
        target_jp.CreateAttribute(
            f"physxMimicJoint:{axis_token}:naturalFrequency", Sdf.ValueTypeNames.Float
        ).Set(natural_frequency)
        target_jp.CreateAttribute(
            f"physxMimicJoint:{axis_token}:dampingRatio", Sdf.ValueTypeNames.Float
        ).Set(damping_ratio)
        configured.append({
            "target_joint_path": str(target_jp.GetPath()),
            "reference_joint_path": str(source_jp.GetPath()),
            "gearing": float(gearing), "offset": float(offset),
        })
    return configured


def _apply_collision_friction(
    physics_usd: Path | None, static_friction: float, dynamic_friction: float, restitution: float,
) -> dict:
    """Apply friction attrs to collision meshes in the *physics* sublayer."""
    from pxr import Sdf, Usd, UsdPhysics
    if physics_usd is None:
        return {"applied_collision_materials": 0, "warning": "physics sublayer not found"}
    stage = Usd.Stage.Open(str(physics_usd))
    applied: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float, False).Set(static_friction)
        prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float, False).Set(dynamic_friction)
        prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float, False).Set(restitution)
        applied.append(path)
    stage.GetRootLayer().Save()
    return {"applied_collision_materials": len(applied)}


def _apply_contact_excludes(stage, contact_excludes: list[dict]) -> dict:
    """Add MJCF contact-exclude pairs to USD FilteredPairsAPI on rigid bodies."""
    from pxr import UsdPhysics
    if not contact_excludes:
        return {"applied_filtered_pairs": 0, "skipped_filtered_pairs": 0, "details": []}
    body_name_to_prim = _find_rigid_body_prims(stage)
    applied = 0
    skipped = 0
    details: list[dict[str, object]] = []
    for rule in contact_excludes:
        p1 = body_name_to_prim.get(rule["body1"])
        p2 = body_name_to_prim.get(rule["body2"])
        if p1 is None or p2 is None:
            skipped += 1
            details.append(
                {
                    "name": rule.get("name", ""),
                    "body1": rule["body1"],
                    "body2": rule["body2"],
                    "body1_prim_path": str(p1.GetPath()) if p1 is not None else None,
                    "body2_prim_path": str(p2.GetPath()) if p2 is not None else None,
                    "status": "skipped",
                }
            )
            continue
        UsdPhysics.FilteredPairsAPI(p1).GetFilteredPairsRel().AddTarget(p2.GetPath())
        applied += 1
        details.append(
            {
                "name": rule.get("name", ""),
                "body1": rule["body1"],
                "body2": rule["body2"],
                "body1_prim_path": str(p1.GetPath()),
                "body2_prim_path": str(p2.GetPath()),
                "status": "applied",
            }
        )
    return {"applied_filtered_pairs": applied, "skipped_filtered_pairs": skipped, "details": details}


def _apply_visual_rgba(base_usd: Path | None, geom_rgba: dict[str, list[float]]) -> dict:
    """Clear white DefaultMaterial bindings and set displayColor on visual meshes."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    if not geom_rgba:
        return {"visual_rgba_applied": 0}
    if base_usd is None:
        return {"visual_rgba_applied": 0, "warning": "base sublayer not found"}
    body_rgba = {k: tuple(v) for k, v in geom_rgba.items()}
    mesh_rgba: dict[str, tuple[float, float, float, float]] = {}
    for body_name, rgba in body_rgba.items():
        mesh_rgba[_strip_mesh_prefix(body_name)] = rgba
    stage = Usd.Stage.Open(str(base_usd))
    applied = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        path_parts = str(prim.GetPath()).split("/")
        if len(path_parts) < 3 or path_parts[1] not in ("meshes", "visuals"):
            continue
        mesh_name = _strip_mesh_prefix(path_parts[2])
        rgba = mesh_rgba.get(mesh_name)
        if rgba is None:
            continue
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().ClearTargets(True)
        UsdGeom.Mesh(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
        applied += 1
    stage.GetRootLayer().Save()
    return {"visual_rgba_applied": applied}


# ---------------------------------------------------------------------------
# Post-conversion summary (read back from the converted USD)
# ---------------------------------------------------------------------------

def _collect_articulation_summary(usd_path: Path) -> dict:
    """Read the converted USD via sublayers and return key articulation data."""
    from pxr import Usd, UsdPhysics

    usd_dir = usd_path.parent

    # Joint data lives in the physics sublayer.
    physics_usd = _find_sublayer_path(usd_dir, "*_physics.usd")
    if physics_usd is None:
        return {"error": "physics sublayer not found", "num_joints": 0, "num_actuated": 0,
                "num_mimic": 0, "num_bodies": 0, "joints": [], "body_names": [],
                "gravity": None, "articulation_root": None}
    phys_stage = Usd.Stage.Open(str(physics_usd))

    joints = []
    for prim in phys_stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        info: dict = {"name": prim.GetName(), "path": str(prim.GetPath()),
                       "type": str(prim.GetTypeName())}
        jtype = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, jtype)
        if drive:
            try:
                info["drive_stiffness"] = drive.GetStiffnessAttr().Get()
                info["drive_damping"] = drive.GetDampingAttr().Get()
                info["drive_max_force"] = drive.GetMaxForceAttr().Get()
            except Exception:
                pass
        try:
            info["armature"] = prim.GetAttribute("physxJoint:armature").Get()
        except Exception:
            pass
        try:
            info["limit_low"] = prim.GetAttribute("physxLimit:X:low").Get()
            info["limit_high"] = prim.GetAttribute("physxLimit:X:high").Get()
        except Exception:
            pass
        info["has_mimic"] = any("Mimic" in a for a in (prim.GetAppliedSchemas() or []))
        joints.append(info)

    # Bodies and articulation root from the composed main stage.
    main_stage = Usd.Stage.Open(str(usd_path))
    art_root = None
    for prim in main_stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = prim
            break
    bodies = sorted(prim.GetName() for prim in main_stage.Traverse()
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI))

    return {
        "num_joints": len(joints),
        "num_actuated": sum(1 for j in joints if j.get("drive_stiffness") not in (None, 0.0)),
        "num_mimic": sum(1 for j in joints if j.get("has_mimic")),
        "num_bodies": len(bodies),
        "body_names": bodies,
        "joints": joints,
        "gravity": None,
        "articulation_root": str(art_root.GetPath()) if art_root else None,
    }


# ---------------------------------------------------------------------------
# Master postprocess
# ---------------------------------------------------------------------------

def _postprocess_hand_usd(usd_path: Path, profile: dict, contact_excludes: list[dict], geom_rgba: dict[str, list[float]]) -> dict:
    """Run only sublayer-safe postprocess steps.

    Do not open+save the composed root USD here. Authoring overrides onto the
    root layer changes runtime behavior for imported hands.
    """
    usd_dir = usd_path.parent
    root_usd = usd_path
    base_usd = _find_sublayer_path(usd_dir, "*_base.usd")
    physics_usd = _find_sublayer_path(usd_dir, "*_physics.usd")
    result: dict = {}

    result["articulation_root_cleanup"] = _apply_articulation_root_cleanup(
        root_usd, physics_usd, preferred_suffix=profile.get("articulation_root_suffix")
    )
    result["worldbody_cleanup"] = _apply_worldbody_cleanup(root_usd, base_usd, physics_usd)

    helper_descendant_names = tuple(profile.get("helper_body_names", ())) + tuple(
        profile.get("helper_container_names", ())
    )

    if base_usd is None:
        result["apply_helper_descendant_cleanup_base"] = []
        result["visual_rgba"] = {"visual_rgba_applied": 0, "warning": "base sublayer not found"}
    else:
        from pxr import Usd

        base_stage = Usd.Stage.Open(str(base_usd))
        result["apply_helper_descendant_cleanup_base"] = _apply_helper_descendant_cleanup(
            base_stage, helper_descendant_names
        )
        base_stage.GetRootLayer().Save()
        result["visual_rgba"] = _apply_visual_rgba(base_usd, geom_rgba)

    if physics_usd is None:
        result["apply_helper_descendant_cleanup_physics"] = []
        result["root_body_mass_fix"] = {"applied": False, "reason": "physics sublayer not found"}
        result["collision_friction"] = {"applied_collision_materials": 0, "warning": "physics sublayer not found"}
        result["apply_drive_settings"] = []
        result["apply_mimic_joints"] = []
        result["contact_excludes"] = {
            "applied_filtered_pairs": 0,
            "skipped_filtered_pairs": len(contact_excludes),
            "details": [],
        }
        result["apply_helper_body_cleanup"] = []
    else:
        from pxr import Usd

        physics_stage = Usd.Stage.Open(str(physics_usd))
        result["apply_helper_descendant_cleanup_physics"] = _apply_helper_descendant_cleanup(
            physics_stage, helper_descendant_names
        )
        result["root_body_mass_fix"] = _apply_root_body_mass_fix(
            physics_stage, profile.get("articulation_root_suffix")
        )
        physics_stage.GetRootLayer().Save()
        result["collision_friction"] = _apply_collision_friction(
            physics_usd,
            profile["contact_friction_static"],
            profile["contact_friction_dynamic"],
            profile["contact_restitution"],
        )
        physics_stage = Usd.Stage.Open(str(physics_usd))
        result["apply_drive_settings"] = _apply_drive_settings(
            physics_stage,
            profile["actuated_joint_names"],
            profile["drive_stiffness"],
            profile["drive_damping"],
            profile["drive_max_force"],
        )
        mimic_specs = profile.get("mimic_linear_specs", ())
        result["apply_mimic_joints"] = (
            _apply_mimic_joints(
                physics_stage,
                mimic_specs,
                profile["mimic_natural_frequency"],
                profile["mimic_damping_ratio"],
            )
            if mimic_specs
            else []
        )
        result["contact_excludes"] = _apply_contact_excludes(physics_stage, contact_excludes)
        result["apply_helper_body_cleanup"] = _apply_helper_body_cleanup(
            physics_stage, profile["helper_body_names"]
        )
        physics_stage.GetRootLayer().Save()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    xml_path, output_dir, profile = _resolve_paths(args)
    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {xml_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _parse_mjcf_metadata(xml_path)
    usd_path, sim_app = _convert_mjcf(xml_path, output_dir, args.fix_base, args.make_instanceable)

    try:
        try:
            metadata["usd_postprocess"] = _postprocess_hand_usd(
                usd_path, profile,
                contact_excludes=metadata.get("contact_excludes", []),
                geom_rgba=metadata.get("geom_rgba", {}),
            )
        except Exception as pp_exc:
            import traceback as _tb
            crash_path = output_dir / "POSTPROCESS_CRASH.txt"
            crash_path.write_text(f"Postprocess crashed: {pp_exc}\n\n{_tb.format_exc()}\n", encoding="utf-8")
            raise
        metadata["usd_path"] = str(usd_path)
        metadata["output_dir"] = str(output_dir)
        metadata["hand_profile"] = args.hand
        metadata["import_options"] = {
            "fix_base": args.fix_base,
            "self_collision": True,
            "make_instanceable": args.make_instanceable,
            "create_physics_scene": False,
        }
        try:
            summary = _collect_articulation_summary(usd_path)
        except Exception as exc:
            import traceback as _tb2
            summary = {"error": f"{exc}\n{_tb2.format_exc()}", "num_joints": 0,
                       "num_actuated": 0, "num_mimic": 0, "num_bodies": 0,
                       "articulation_root": None, "gravity": None, "joints": []}
        metadata["articulation_summary"] = summary

        try:
            lines = [f"hand: {args.hand}", f"usd: {usd_path}"]
            lines.append(
                f"joints: {summary.get('num_joints', 0)} total | "
                f"{summary.get('num_actuated', 0)} actuated | "
                f"{summary.get('num_mimic', 0)} mimic"
            )
            lines.append(f"bodies: {summary.get('num_bodies', 0)}")
            lines.append(f"articulation root: {summary.get('articulation_root', '?')}")
            if summary.get("error"):
                lines.append(f"ERROR: {summary['error']}")
            lines.append("done.")
            summary_text = "\n".join(lines)
            print(summary_text, flush=True)
            (output_dir / "conversion_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
        except Exception as sum_exc:
            import traceback as _tb3
            err_path = output_dir / "SUMMARY_ERROR.txt"
            err_path.write_text(f"Summary generation crashed: {sum_exc}\n\n{_tb3.format_exc()}\n", encoding="utf-8")

        report_path = output_dir / "mjcf_migration_report.json"
        report_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
