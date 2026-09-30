"""Visualize a fixed BPS basis with Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - visualization helper
    raise ModuleNotFoundError(
        "open3d is required for visualize_bps_basis.py. "
        "Run this script in an environment with open3d installed."
    ) from exc

try:
    from .bps import default_bps_basis_path
except ImportError:  # pragma: no cover - direct script execution fallback
    from dynamic_dexgrasp_lab.model.backbone.bps import default_bps_basis_path  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a fixed BPS basis file with Open3D.")
    parser.add_argument("--basis-path", type=Path, default=None)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--show-frame", action="store_true")
    parser.add_argument("--frame-size", type=float, default=0.1)
    parser.add_argument("--show-sphere", action="store_true")
    parser.add_argument("--sphere-radius", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basis_path = args.basis_path
    if basis_path is None:
        basis_path = default_bps_basis_path()
    basis_path = Path(basis_path).expanduser().resolve()
    if not basis_path.exists():
        raise FileNotFoundError(f"BPS basis file not found: {basis_path}")

    basis = np.load(basis_path).astype(np.float32)
    if basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError(f"BPS basis must have shape [K, 3], got {basis.shape}.")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(basis)
    point_cloud.paint_uniform_color([0.1, 0.45, 0.95])

    geometries: list[o3d.geometry.Geometry] = [point_cloud]

    if args.show_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.frame_size)))

    if args.show_sphere:
        sphere_radius = args.sphere_radius
        if sphere_radius is None:
            sphere_radius = float(np.linalg.norm(basis, axis=1).max(initial=0.0))
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(sphere_radius), resolution=32)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.85, 0.85, 0.85])
        sphere.compute_triangle_normals()
        geometries.append(sphere)

    print(f"basis_path={basis_path}")
    print(f"num_points={basis.shape[0]}")
    print(f"radius_max={float(np.linalg.norm(basis, axis=1).max(initial=0.0)):.6f}")
    print(f"min={basis.min(axis=0)}")
    print(f"max={basis.max(axis=0)}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="BPS Basis", width=1280, height=960)
    for geometry in geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
