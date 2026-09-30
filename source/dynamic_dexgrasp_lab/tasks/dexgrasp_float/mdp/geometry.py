"""Geometry helpers for heterogeneous object point clouds."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from dynamic_dexgrasp_lab.model.backbone.bps import encode_point_cloud_bps_dists

LOGGER = logging.getLogger(__name__)

_OBJECT_SURFACE_CACHE: dict[tuple[str, int], np.ndarray] = {}
_OBJECT_FULL_SURFACE_CACHE: dict[str, np.ndarray] = {}
_SOURCE_TYPE_CACHE: dict[str, str] = {}


#
# Cache lifecycle
#


def clear_surface_point_cache() -> None:
    """Clear cached object local-frame point clouds."""
    _OBJECT_SURFACE_CACHE.clear()
    _OBJECT_FULL_SURFACE_CACHE.clear()
    _SOURCE_TYPE_CACHE.clear()


#
# Generic helpers
#


def _stable_seed(object_key: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{object_key}|{base_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _stack_point_clouds(points_list: list[torch.Tensor], num_points: int) -> torch.Tensor:
    """Stack a list of same-shaped point clouds into ``(N, P, 3)``."""
    if len(points_list) == 0:
        raise ValueError("points_list must not be empty.")
    if any(points.shape != (num_points, 3) for points in points_list):
        raise ValueError("All point clouds must have shape (num_points, 3) before stacking.")
    return torch.stack(points_list, dim=0)


def _transform_points_b_to_w(
    object_points_b: list[torch.Tensor],
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> list[torch.Tensor]:
    """Transform per-env object-frame point clouds to world frame."""
    points_world_list: list[torch.Tensor] = []
    for env_id, points_b in enumerate(object_points_b):
        repeated_quat = object_quat_w[env_id].unsqueeze(0).expand(points_b.shape[0], -1)
        points_world = math_utils.quat_apply(repeated_quat, points_b) + object_pos_w[env_id].unsqueeze(0)
        points_world_list.append(points_world)
    return points_world_list


def _transform_points_b_to_w_batched(
    object_points_b: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform batched object-frame point clouds ``(N, P, 3)`` to world frame."""
    repeated_quat = object_quat_w.unsqueeze(1).expand(-1, object_points_b.shape[1], -1)
    return math_utils.quat_apply(repeated_quat, object_points_b) + object_pos_w.unsqueeze(1)


def _transform_points_w_to_frame(
    points_w_list: list[torch.Tensor],
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> list[torch.Tensor]:
    """Transform per-env world-frame point clouds into another frame."""
    points_frame_list: list[torch.Tensor] = []
    for env_id, points_w in enumerate(points_w_list):
        repeated_quat = frame_quat_w[env_id].unsqueeze(0).expand(points_w.shape[0], -1)
        points_frame = math_utils.quat_apply_inverse(
            repeated_quat,
            points_w - frame_pos_w[env_id].unsqueeze(0),
        )
        points_frame_list.append(points_frame)
    return points_frame_list


def _transform_points_w_to_frame_batched(
    points_w: torch.Tensor,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform batched world-frame point clouds ``(N, P, 3)`` into another frame."""
    repeated_quat = frame_quat_w.unsqueeze(1).expand(-1, points_w.shape[1], -1)
    return math_utils.quat_apply_inverse(repeated_quat, points_w - frame_pos_w.unsqueeze(1))


def _transform_points_frame_to_w(
    points_frame_list: list[torch.Tensor],
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> list[torch.Tensor]:
    """Transform per-env point clouds from a local frame to world."""
    points_w_list: list[torch.Tensor] = []
    for env_id, points_frame in enumerate(points_frame_list):
        repeated_quat = frame_quat_w[env_id].unsqueeze(0).expand(points_frame.shape[0], -1)
        points_w = math_utils.quat_apply(repeated_quat, points_frame) + frame_pos_w[env_id].unsqueeze(0)
        points_w_list.append(points_w)
    return points_w_list


def _transform_points_frame_to_w_batched(
    points_frame: torch.Tensor,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform batched local-frame point clouds ``(N, P, 3)`` to world frame."""
    repeated_quat = frame_quat_w.unsqueeze(1).expand(-1, points_frame.shape[1], -1)
    return math_utils.quat_apply(repeated_quat, points_frame) + frame_pos_w.unsqueeze(1)


def _ensure_camera_home_pose(env) -> None:
    """Ensure the lightweight virtual camera state exists."""
    if hasattr(env, "dexgrasp_cam_pos_w") and hasattr(env, "dexgrasp_cam_quat_w"):
        return

    camera_object_relative_position = getattr(env.cfg, "student_camera_object_relative_position", None)
    camera_object_relative_quat = getattr(env.cfg, "student_camera_object_relative_quat", None)
    if camera_object_relative_position is None or camera_object_relative_quat is None:
        raise RuntimeError(
            "Student virtual camera pose is unavailable and no default camera-relative pose is configured."
        )

    env.dexgrasp_cam_pos_w = (
        torch.tensor(camera_object_relative_position, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(env.num_envs, -1)
        + env.scene["object"].data.root_pos_w
    )
    env.dexgrasp_cam_quat_w = math_utils.quat_unique(
        torch.tensor(camera_object_relative_quat, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(env.num_envs, -1)
    )


#
# Unified resampling and visibility
#


def farthest_point_sampling(points: torch.Tensor, n_samples: int, memory_threshold: int = 2 * 1024**3) -> torch.Tensor:
    """Farthest point sampling for a single point set."""
    device = points.device
    num_points = points.shape[0]
    if num_points <= n_samples:
        return torch.arange(num_points, device=device, dtype=torch.long)[:n_samples]

    bytes_needed = num_points * num_points * points.element_size()
    if bytes_needed <= memory_threshold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((num_points,), float("inf"), device=device)
        farthest = torch.tensor(0, device=device, dtype=torch.long)
        for step in range(n_samples):
            sampled_idx[step] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx

    logging.warning("FPS fallback to iterative mode for %d points.", num_points)
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((num_points,), float("inf"), device=device)
    farthest = torch.tensor(0, device=device, dtype=torch.long)
    for step in range(n_samples):
        sampled_idx[step] = farthest
        dist = torch.linalg.norm(points - points[farthest], dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def random_point_sampling(
    points: torch.Tensor,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random point sampling indices for a single point set."""
    num_points = points.shape[0]
    device = points.device
    if num_points <= n_samples:
        return torch.arange(num_points, device=device, dtype=torch.long)[:n_samples]
    return torch.randperm(num_points, generator=generator, device=device)[:n_samples]


def _resample_points_deterministic_numpy(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Match the old deterministic full-sample contract exactly via numpy choice."""
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < num_points
    indices = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def obj_pc_resample(
    points: torch.Tensor,
    num_points: int,
    *,
    method: str = "fps",
    deterministic: bool = False,
    seed: int | None = None,
    max_candidates: int | None = 4096,
) -> torch.Tensor:
    """Unified point-cloud resampling helper."""
    device = points.device
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if method not in {"fps", "random"}:
        raise ValueError(f"Unsupported obj_pc_resample method: {method!r}")
    if points.shape[0] == 0:
        return torch.zeros((num_points, 3), dtype=torch.float32, device=device)

    generator = None
    if deterministic:
        generator = torch.Generator(device=device)
        generator.manual_seed(0 if seed is None else int(seed))

    if max_candidates is not None and points.shape[0] > max_candidates:
        keep = torch.randperm(points.shape[0], generator=generator, device=device)[:max_candidates]
        points = points[keep]

    if points.shape[0] >= num_points:
        if method == "fps":
            return points[farthest_point_sampling(points, num_points)]
        keep = random_point_sampling(points, num_points, generator=generator)
        return points[keep]

    extra = torch.randint(
        0,
        points.shape[0],
        (num_points - points.shape[0],),
        generator=generator,
        device=device,
    )
    base_idx = torch.arange(points.shape[0], device=device)
    return points[torch.cat([base_idx, extra], dim=0)]


def _project_visible_points(
    points_camera: torch.Tensor,
    *,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    near: float,
    far: float,
    depth_epsilon: float,
) -> torch.Tensor:
    """Apply frustum crop and approximate z-buffer visibility in camera frame."""
    if points_camera.shape[0] == 0:
        return points_camera

    z = points_camera[:, 2]
    valid_depth = (z > near) & (z < far)
    if not torch.any(valid_depth):
        return points_camera[:0]

    points_camera = points_camera[valid_depth]
    z = points_camera[:, 2]

    hfov_rad = math.radians(float(horizontal_fov_deg))
    focal_px = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
    focal_py = focal_px
    cx = float(width) * 0.5
    cy = float(height) * 0.5

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    u = focal_px * (x / z) + cx
    v = focal_py * (y / z) + cy

    valid_fov = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    if not torch.any(valid_fov):
        return points_camera[:0]

    points_camera = points_camera[valid_fov]
    z = z[valid_fov]
    u = u[valid_fov].to(dtype=torch.long)
    v = v[valid_fov].to(dtype=torch.long)

    flat_idx = v * width + u
    min_depth = torch.full((width * height,), float("inf"), dtype=torch.float32, device=points_camera.device)
    min_depth.scatter_reduce_(0, flat_idx, z, reduce="amin", include_self=True)
    visible = z <= (min_depth[flat_idx] + float(depth_epsilon))
    return points_camera[visible]


def _project_visible_mask_batched(
    points_camera: torch.Tensor,
    *,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    near: float,
    far: float,
    depth_epsilon: float,
) -> torch.Tensor:
    """Apply frustum crop and approximate z-buffer visibility for ``(N, P, 3)`` camera-frame points."""
    num_envs, num_points, _ = points_camera.shape
    if num_points == 0:
        return torch.zeros((num_envs, 0), dtype=torch.bool, device=points_camera.device)

    z = points_camera[:, :, 2]
    valid_depth = (z > near) & (z < far)

    hfov_rad = math.radians(float(horizontal_fov_deg))
    focal_px = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
    focal_py = focal_px
    cx = float(width) * 0.5
    cy = float(height) * 0.5

    z_safe = torch.where(valid_depth, z, torch.ones_like(z))
    x = points_camera[:, :, 0]
    y = points_camera[:, :, 1]
    u = focal_px * (x / z_safe) + cx
    v = focal_py * (y / z_safe) + cy

    valid_fov = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    valid = valid_depth & valid_fov
    if not torch.any(valid):
        return torch.zeros((num_envs, num_points), dtype=torch.bool, device=points_camera.device)

    env_ids = torch.arange(num_envs, device=points_camera.device, dtype=torch.long).unsqueeze(1).expand(-1, num_points)
    flat_env = env_ids[valid]
    flat_u = u[valid].to(dtype=torch.long)
    flat_v = v[valid].to(dtype=torch.long)
    flat_z = z[valid]

    width_height = width * height
    flat_idx = flat_v * width + flat_u
    global_idx = flat_env * width_height + flat_idx

    min_depth = torch.full(
        (num_envs * width_height,),
        float("inf"),
        dtype=torch.float32,
        device=points_camera.device,
    )
    min_depth.scatter_reduce_(0, global_idx, flat_z, reduce="amin", include_self=True)
    visible_flat = flat_z <= (min_depth[global_idx] + float(depth_epsilon))

    visible = torch.zeros((num_envs, num_points), dtype=torch.bool, device=points_camera.device)
    visible[valid] = visible_flat
    return visible


#
# Object source loading
#


def _load_points_from_global_pc(path: str) -> np.ndarray:
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"global_pc.npy must be (N, 3), got {points.shape} at {path}")
    points = points.astype(np.float32, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        raise ValueError(f"global_pc.npy has no finite points: {path}")
    return points


def _triangulate_faces(prim) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
    return np.asarray(faces, dtype=np.int64)


def _create_trimesh_for_prim(prim):
    import trimesh
    from pxr import UsdGeom

    prim_type = prim.GetTypeName()
    if prim_type == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    if prim_type == "Sphere":
        radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=radius)
    if prim_type == "Cylinder":
        cyl = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=cyl.GetRadiusAttr().Get(), height=cyl.GetHeightAttr().Get())
    if prim_type == "Capsule":
        cap = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(radius=cap.GetRadiusAttr().Get(), height=cap.GetHeightAttr().Get())
    if prim_type == "Cone":
        cone = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=cone.GetRadiusAttr().Get(), height=cone.GetHeightAttr().Get())
    raise KeyError(f"Unsupported primitive type for surface sampling: {prim_type}")


def _sample_points_from_stage_object(*, usd_path: str, prim_path: str, num_points: int) -> np.ndarray:
    import trimesh
    from pxr import UsdGeom

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    if stage is None:
        raise RuntimeError("USD sampling fallback requires an active stage.")
    if usd_path:
        matched_paths: list[str] = []
        for prim in stage.Traverse():
            refs = prim.GetMetadata("references")
            if refs is None:
                continue
            if usd_path in str(refs):
                matched_paths.append(str(prim.GetPath()))
        if len(matched_paths) == 0:
            raise RuntimeError(f"Unable to find prim on stage for usd fallback: {usd_path}")
        prim_path = matched_paths[0]
    if not prim_path:
        raise RuntimeError("Stage object sampling requires either a non-empty usd_path or object prim_path.")

    xform_cache = UsdGeom.XformCache()
    object_prim = stage.GetPrimAtPath(prim_path)
    supported_types = ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
    prims = []
    if object_prim.GetTypeName() in supported_types:
        prims.append(object_prim)
    prims.extend(
        sim_utils.get_all_matching_child_prims(
            prim_path,
            predicate=lambda prim: prim.GetTypeName() in supported_types,
        )
    )
    if not prims:
        raise KeyError(f"No valid geometry prims found under {prim_path}.")

    world_root = xform_cache.GetLocalToWorldTransform(object_prim)
    sampled_parts: list[np.ndarray] = []
    for prim in prims:
        mesh_tm = _create_trimesh_for_prim(prim)
        dense_samples, _ = trimesh.sample.sample_surface(mesh_tm, num_points, face_weight=mesh_tm.area_faces)
        local_pts = torch.from_numpy(dense_samples.astype(np.float32))
        rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
        mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
        mat_t = torch.from_numpy(mat_np)
        ones = torch.ones((local_pts.shape[0], 1), dtype=torch.float32)
        pts_h = torch.cat([local_pts, ones], dim=-1)
        root_h = pts_h @ mat_t
        sampled_parts.append(root_h[:, :3].detach().cpu().numpy())
    if len(sampled_parts) == 1:
        return sampled_parts[0].astype(np.float32)
    return np.concatenate(sampled_parts, axis=0).astype(np.float32)


def _load_object_full_surface_points_local(
    *,
    object_key: str,
    usd_path: str,
    prim_path: str,
    dataset_pc_path: str | None,
    device: str,
    point_source: str,
) -> torch.Tensor:
    if object_key in _OBJECT_FULL_SURFACE_CACHE:
        return torch.from_numpy(_OBJECT_FULL_SURFACE_CACHE[object_key]).to(device).clone()

    prefer_dataset_pc = point_source in {"global_pc", "dataset_or_usd"}
    if prefer_dataset_pc and dataset_pc_path is not None and Path(dataset_pc_path).is_file():
        points_np = _load_points_from_global_pc(dataset_pc_path)
        source_type = "dataset_pc"
    else:
        points_np = _sample_points_from_stage_object(
            usd_path=usd_path,
            prim_path=prim_path,
            num_points=4096,
        )
        source_type = "usd"

    _OBJECT_FULL_SURFACE_CACHE[object_key] = points_np.astype(np.float32, copy=False)
    _SOURCE_TYPE_CACHE[object_key] = source_type
    LOGGER.info("Loaded full object surface points: object=%s source=%s", object_key, source_type)
    return torch.from_numpy(_OBJECT_FULL_SURFACE_CACHE[object_key]).to(device).clone()


def _resolve_object_point_cloud_metadata(
    env,
    *,
    device: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str | None, ...], tuple[str, ...], str, str]:
    """Resolve per-env object point-cloud metadata with single-object fallback."""
    object_keys = getattr(env.cfg, "_env_object_scale_keys", None)
    object_usd_paths = getattr(env.cfg, "_env_object_usd_paths", None)
    object_dataset_pc_paths = getattr(env.cfg, "_env_object_dataset_pc_paths", None)
    if object_dataset_pc_paths is None:
        object_dataset_pc_paths = getattr(env.cfg, "_env_object_global_pc_paths", None)

    obj = env.scene["object"]
    object_prim_paths = tuple(obj.root_physx_view.prim_paths[: env.num_envs])
    if object_keys is None or object_usd_paths is None or object_dataset_pc_paths is None:
        object_keys = tuple(f"default:{object_prim_paths[0]}" for _ in range(env.num_envs))
        object_usd_paths = tuple("" for _ in range(env.num_envs))
        object_dataset_pc_paths = tuple(None for _ in range(env.num_envs))

    if not (len(object_keys) == len(object_usd_paths) == len(object_dataset_pc_paths) == env.num_envs):
        raise ValueError("env object assignment metadata shape mismatch with num_envs.")

    point_source = str(getattr(env.cfg, "geometry_point_cloud_source", "usd")).strip().lower()
    if point_source not in {"usd", "global_pc", "dataset_or_usd"}:
        raise ValueError(f"Unsupported geometry_point_cloud_source: {point_source!r}")

    target_device = str(device or env.device)
    return object_keys, object_usd_paths, object_dataset_pc_paths, object_prim_paths, point_source, target_device


#
# Public full/reference point clouds
#


def obj_pc_full_b(env, *, device: str | None = None) -> torch.Tensor:
    """Return per-env full object point clouds in object/body frame."""
    object_keys, object_usd_paths, object_dataset_pc_paths, object_prim_paths, point_source, target_device = (
        _resolve_object_point_cloud_metadata(env, device=device)
    )
    unique_keys = tuple(dict.fromkeys(object_keys))
    first_env_idx: dict[str, int] = {}
    for env_idx, object_key in enumerate(object_keys):
        first_env_idx.setdefault(object_key, env_idx)

    key_to_tensor: dict[str, torch.Tensor] = {}
    for object_key in unique_keys:
        idx = first_env_idx[object_key]
        key_to_tensor[object_key] = _load_object_full_surface_points_local(
            object_key=object_key,
            usd_path=object_usd_paths[idx],
            prim_path=object_prim_paths[idx],
            dataset_pc_path=object_dataset_pc_paths[idx],
            device=target_device,
            point_source=point_source,
        )
    return torch.stack([key_to_tensor[key].clone() for key in object_keys], dim=0)


def obj_pc_full_w(env) -> torch.Tensor:
    """Return per-env full object point clouds in world frame."""
    obj = env.scene["object"]
    return _transform_points_b_to_w_batched(
        obj_pc_full_b(env),
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )


def obj_pc_full_h(env) -> torch.Tensor:
    """Return per-env full object point clouds in hand-root frame."""
    robot = env.scene["robot"]
    return _transform_points_w_to_frame_batched(
        obj_pc_full_w(env),
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


def obj_pc_full_c(env) -> torch.Tensor:
    """Return per-env full object point clouds in virtual camera frame."""
    _ensure_camera_home_pose(env)
    return _transform_points_w_to_frame_batched(
        obj_pc_full_w(env),
        env.dexgrasp_cam_pos_w,
        env.dexgrasp_cam_quat_w,
    )


def obj_pc_full_sample_b(env, num_points: int) -> torch.Tensor:
    """Return per-env deterministic sampled full point clouds in object/body frame."""
    object_keys, object_usd_paths, object_dataset_pc_paths, object_prim_paths, point_source, _ = (
        _resolve_object_point_cloud_metadata(env)
    )
    unique_keys = tuple(dict.fromkeys(object_keys))
    first_env_idx: dict[str, int] = {}
    for env_idx, object_key in enumerate(object_keys):
        first_env_idx.setdefault(object_key, env_idx)

    keys_to_load = tuple(key for key in unique_keys if (key, num_points) not in _OBJECT_SURFACE_CACHE)
    if len(keys_to_load) > 0:
        msg = (
            f"[GEOM] Preparing object point clouds: source={point_source} "
            f"num_objects={len(keys_to_load)} num_points={num_points}"
        )
        LOGGER.info(msg)
        print(msg, flush=True)
        prepare_start = time.perf_counter()

    key_to_tensor: dict[str, torch.Tensor] = {}
    for object_key in unique_keys:
        cache_key = (object_key, num_points)
        if cache_key not in _OBJECT_SURFACE_CACHE:
            idx = first_env_idx[object_key]
            _load_object_full_surface_points_local(
                object_key=object_key,
                usd_path=object_usd_paths[idx],
                prim_path=object_prim_paths[idx],
                dataset_pc_path=object_dataset_pc_paths[idx],
                device=env.device,
                point_source=point_source,
            )
            object_seed = _stable_seed(
                object_key,
                int(env.cfg.seed) if getattr(env.cfg, "seed", None) is not None else 0,
            )
            sampled = _resample_points_deterministic_numpy(
                _OBJECT_FULL_SURFACE_CACHE[object_key],
                num_points=num_points,
                seed=object_seed,
            )
            _OBJECT_SURFACE_CACHE[cache_key] = sampled
            LOGGER.info(
                "Loaded object full sample points: object=%s source=%s",
                object_key,
                _SOURCE_TYPE_CACHE[object_key],
            )
        key_to_tensor[object_key] = torch.from_numpy(_OBJECT_SURFACE_CACHE[cache_key]).to(env.device).clone()

    if len(keys_to_load) > 0:
        elapsed = time.perf_counter() - prepare_start
        msg = f"[GEOM] Point cloud preparation done in {elapsed:.2f}s"
        LOGGER.info(msg)
        print(msg, flush=True)

    return torch.stack([key_to_tensor[key] for key in object_keys], dim=0)


def obj_pc_full_sample_w(env, num_points: int) -> torch.Tensor:
    """Return per-env deterministic sampled full point clouds in world frame."""
    obj = env.scene["object"]
    return _transform_points_b_to_w_batched(
        obj_pc_full_sample_b(env, num_points),
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )


def obj_pc_full_sample_h(env, num_points: int) -> torch.Tensor:
    """Return per-env deterministic sampled full point clouds in hand-root frame."""
    robot = env.scene["robot"]
    return _transform_points_w_to_frame_batched(
        obj_pc_full_sample_w(env, num_points),
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


def obj_pc_full_sample_c(env, num_points: int) -> torch.Tensor:
    """Return per-env deterministic sampled full point clouds in virtual camera frame."""
    _ensure_camera_home_pose(env)
    return _transform_points_w_to_frame_batched(
        obj_pc_full_sample_w(env, num_points),
        env.dexgrasp_cam_pos_w,
        env.dexgrasp_cam_quat_w,
    )


def obj_pc_part_c(env) -> list[torch.Tensor]:
    """Return raw partial point clouds in camera frame."""
    full_points_b = obj_pc_full_b(env)
    obj = env.scene["object"]
    full_points_w = _transform_points_b_to_w_batched(full_points_b, obj.data.root_pos_w, obj.data.root_quat_w)
    _ensure_camera_home_pose(env)
    full_points_c = _transform_points_w_to_frame_batched(full_points_w, env.dexgrasp_cam_pos_w, env.dexgrasp_cam_quat_w)

    width = int(getattr(env.cfg, "student_zbuf_width", 64))
    height = int(getattr(env.cfg, "student_zbuf_height", 48))
    horizontal_fov_deg = float(getattr(env.cfg, "student_camera_horizontal_fov_deg", 70.0))
    near = float(getattr(env.cfg, "student_camera_near", 0.05))
    far = float(getattr(env.cfg, "student_camera_far", 2.0))
    depth_epsilon = float(getattr(env.cfg, "student_visibility_depth_margin", 0.005))
    noise_std = float(getattr(env.cfg, "student_partial_pc_noise_std", 0.0))

    visible_mask = _project_visible_mask_batched(
        full_points_c,
        width=width,
        height=height,
        horizontal_fov_deg=horizontal_fov_deg,
        near=near,
        far=far,
        depth_epsilon=depth_epsilon,
    )
    if noise_std > 0.0:
        full_points_c = full_points_c + noise_std * torch.randn_like(full_points_c)

    return [full_points_c[env_id][visible_mask[env_id]].clone() for env_id in range(env.num_envs)]


def obj_pc_part_w(env) -> list[torch.Tensor]:
    """Return raw partial point clouds in world frame."""
    _ensure_camera_home_pose(env)
    return _transform_points_frame_to_w(
        obj_pc_part_c(env),
        env.dexgrasp_cam_pos_w,
        env.dexgrasp_cam_quat_w,
    )


def obj_pc_part_h(env) -> list[torch.Tensor]:
    """Return raw partial point clouds in hand-root frame."""
    robot = env.scene["robot"]
    return _transform_points_w_to_frame(
        obj_pc_part_w(env),
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


#
# Sampled partial point clouds
#


def _sample_ragged_point_clouds_random(points_list: list[torch.Tensor], num_points: int) -> torch.Tensor:
    """Randomly sample fixed-size point clouds from ragged per-env inputs without changing semantics."""
    if len(points_list) == 0:
        raise ValueError("points_list must not be empty.")

    device = points_list[0].device
    num_envs = len(points_list)
    lengths = torch.tensor([points.shape[0] for points in points_list], dtype=torch.long, device=device)
    max_points = int(lengths.max().item()) if num_envs > 0 else 0
    if max_points == 0:
        return torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)

    padded = torch.zeros((num_envs, max_points, 3), dtype=torch.float32, device=device)
    for env_id, points in enumerate(points_list):
        point_count = points.shape[0]
        if point_count > 0:
            padded[env_id, :point_count] = points

    sampled_idx = torch.zeros((num_envs, num_points), dtype=torch.long, device=device)
    has_enough = lengths >= num_points
    if torch.any(has_enough):
        scores = torch.rand((num_envs, max_points), device=device)
        valid = torch.arange(max_points, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.where(valid, scores, torch.full_like(scores, float("inf")))
        sampled_idx[has_enough] = torch.topk(scores[has_enough], k=num_points, dim=1, largest=False).indices

    needs_repeat = ~has_enough
    if torch.any(needs_repeat):
        base = torch.arange(num_points, device=device, dtype=torch.long).unsqueeze(0).expand(num_envs, -1)
        tail_mask = base >= lengths.unsqueeze(1)
        safe_lengths = torch.clamp(lengths, min=1).unsqueeze(1)
        extra = torch.floor(torch.rand((num_envs, num_points), device=device) * safe_lengths.float()).to(torch.long)
        sampled_idx[needs_repeat] = torch.where(
            tail_mask[needs_repeat],
            extra[needs_repeat],
            base[needs_repeat],
        )

    return torch.gather(padded, dim=1, index=sampled_idx.unsqueeze(-1).expand(-1, -1, 3))


def obj_pc_part_sample_c(env, num_points: int) -> torch.Tensor:
    """Return sampled partial point clouds in camera frame."""
    step_token = int(getattr(env, "common_step_counter", 0))
    cache_token_name = "_dexgrasp_obj_pc_part_sample_c_step"
    cache_value_name = "_dexgrasp_obj_pc_part_sample_c_value"
    if getattr(env, cache_token_name, None) != step_token:
        setattr(env, cache_value_name, _sample_ragged_point_clouds_random(obj_pc_part_c(env), num_points).clone())
        setattr(env, cache_token_name, step_token)
    return getattr(env, cache_value_name).clone()


def obj_pc_part_sample_w(env, num_points: int) -> torch.Tensor:
    """Return sampled partial point clouds in world frame."""
    _ensure_camera_home_pose(env)
    sampled_world = _transform_points_frame_to_w_batched(
        obj_pc_part_sample_c(env, num_points),
        env.dexgrasp_cam_pos_w,
        env.dexgrasp_cam_quat_w,
    )
    return sampled_world


def obj_pc_part_sample_h(env, num_points: int) -> torch.Tensor:
    """Return sampled partial point clouds in hand-root frame."""
    robot = env.scene["robot"]
    _ensure_camera_home_pose(env)
    sampled_world = _transform_points_frame_to_w_batched(
        obj_pc_part_sample_c(env, num_points),
        env.dexgrasp_cam_pos_w,
        env.dexgrasp_cam_quat_w,
    )
    return _transform_points_w_to_frame_batched(
        sampled_world,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


#
# Observation encoding
#


def obj_pc_part_obs(env, num_points: int = 512) -> torch.Tensor:
    """Return the fixed-size partial-point-cloud observation in hand-root frame."""
    return obj_pc_part_sample_h(env, num_points).reshape(env.num_envs, -1)


def obj_pc_full_bps_obs(env, num_points: int = 512, basis_path: str | Path | None = None) -> torch.Tensor:
    """Return fixed-size BPS distance features from hand-root-frame full sampled point clouds."""
    point_cloud_h = obj_pc_full_sample_h(env, num_points)
    if basis_path is None or not str(basis_path).strip():
        raise ValueError("obj_pc_full_bps_obs requires an explicit basis_path.")
    bps_feature = encode_point_cloud_bps_dists(point_cloud_h, basis_path=basis_path)
    return bps_feature.reshape(env.num_envs, -1)


def obj_pc_part_bps_obs(env, num_points: int = 512, basis_path: str | Path | None = None) -> torch.Tensor:
    """Return fixed-size BPS distance features from hand-root-frame partial point clouds."""
    point_cloud_h = obj_pc_part_sample_h(env, num_points)
    if basis_path is None or not str(basis_path).strip():
        raise ValueError("obj_pc_part_bps_obs requires an explicit basis_path.")
    bps_feature = encode_point_cloud_bps_dists(point_cloud_h, basis_path=basis_path)
    return bps_feature.reshape(env.num_envs, -1)
