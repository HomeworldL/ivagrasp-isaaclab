"""Dataset and object-set loading helpers for dexgrasp assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Literal, TypeAlias

OBJECT_SET_ID_HELP = (
    "Supported forms: 'train@cluster@cluster0_topk1@s120_120', "
    "'subtest@cluster@clusterALL_topkALL@s120_120', "
    "'subood@cluster@clusterALL_topkALL@s120_120', "
    "'ood@object@YCB_077_rubiks_cube@s080_140', and "
    "'train@object_scale@YCB_077_rubiks_cube-scale120'."
)

LOGGER = logging.getLogger(__name__)

SplitSide: TypeAlias = Literal["train", "test", "subtest", "ood", "subood"]
SPLIT_SIDE_PATTERN = "train|test|subtest|ood|subood"


@dataclass(frozen=True)
class SplitObjectAsset:
    """Single object-scale asset entry from one split catalog."""

    object_scale_key: str
    object_name: str
    asset_path: str
    mjcf_path: str
    usd_path: str | None
    scale_tag: str
    scale: float
    cluster_scale_tag: str
    cluster_id: int
    cluster_distance: float
    cluster_rank: int
    distance_to_global_center: float
    cluster_member_index: int

    def resolved_asset_dir(self, dataset_root: str | Path) -> Path:
        return Path(dataset_root).expanduser().resolve() / self.asset_path

    def resolved_mjcf_path(self, dataset_root: str | Path) -> Path:
        return Path(dataset_root).expanduser().resolve() / self.mjcf_path

    def resolved_usd_path(self, dataset_root: str | Path) -> Path:
        dataset_root = Path(dataset_root).expanduser().resolve()
        if self.usd_path:
            return dataset_root / self.usd_path
        return dataset_root / self.asset_path / "object.usd"


@dataclass(frozen=True)
class ClusterMemberMeta:
    """Object-level cluster membership metadata."""

    object_name: str
    cluster_scale_tag: str
    distance_to_center: float
    distance_to_global_center: float
    rank_in_cluster: int
    num_scales: int
    scale_tags: tuple[str, ...]


@dataclass(frozen=True)
class ClusterIndex:
    """Cluster-level ordering metadata."""

    cluster_id: int
    split_center_object_name: str
    distance_to_global_center: float
    member_count: int
    total_scale_count: int
    members: tuple[ClusterMemberMeta, ...]


@dataclass(frozen=True)
class ObjectSelection:
    """Resolved object-set selection for one task config."""

    split_side: SplitSide
    object_set_id: str
    active_assets: tuple[SplitObjectAsset, ...]

    @property
    def active_asset(self) -> SplitObjectAsset:
        if len(self.active_assets) != 1:
            raise ValueError(
                "active_asset is only defined for one-asset selections. "
                f"Got {len(self.active_assets)} assets."
            )
        return self.active_assets[0]


@dataclass
class ObjectSetQueryCfg:
    """Config-first query for object-set selection."""

    dataset_root: str = "/home/ccs/repositories/dexgrasp_sample/datasets/objdata_YCB"
    rl_split_dir: str = (
        "/home/ccs/repositories/dexgrasp_sample/datasets/objdata_YCB/_meta/rl_split/"
        "v1_ae128_k4_seed0_test20_seed0"
    )
    object_set_id: str = "train@cluster@cluster0_topk1@s120_120"


@dataclass(frozen=True)
class RLSplitCatalog:
    """Resolved split assets and cluster indices."""

    split_dir: Path
    side: SplitSide
    assets: tuple[SplitObjectAsset, ...]
    clusters: dict[int, ClusterIndex]

    def asset_by_object_scale_key(self, object_scale_key: str) -> SplitObjectAsset:
        for asset in self.assets:
            if asset.object_scale_key == object_scale_key:
                return asset
        raise KeyError(f"Object-scale key not found in RL split catalog: {object_scale_key}")

    def assets_for_object(self, object_name: str) -> tuple[SplitObjectAsset, ...]:
        matches = [asset for asset in self.assets if asset.object_name == object_name]
        if not matches:
            raise KeyError(f"Object not found in RL split catalog: {object_name}")
        return tuple(sorted(matches, key=lambda asset: (asset.scale, asset.object_scale_key)))

    def all_cluster_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.clusters.keys()))

    @staticmethod
    def _scale_value(scale_tag: str) -> int:
        match = re.fullmatch(r"scale(?P<value>\d+)", scale_tag)
        if match is None:
            raise ValueError(f"Unsupported scale tag format: {scale_tag!r}")
        return int(match.group("value"))

    def _assets_for_object_and_scale_range(
        self,
        object_name: str,
        *,
        scale_min: int,
        scale_max: int,
    ) -> tuple[SplitObjectAsset, ...]:
        matches = [
            asset
            for asset in self.assets_for_object(object_name)
            if scale_min <= self._scale_value(asset.scale_tag) <= scale_max
        ]
        if not matches:
            raise ValueError(
                f"Object {object_name!r} has no assets in scale range "
                f"s{scale_min:03d}_{scale_max:03d}."
            )
        return tuple(matches)

    def _cluster_member_object_names(self, cluster_id: int, *, top_k: int | None) -> tuple[str, ...]:
        cluster = self.clusters[cluster_id]
        if top_k is not None and top_k > len(cluster.members):
            LOGGER.warning(
                "cluster%s_topk%s requested more members than available; truncating to %s.",
                cluster_id,
                top_k,
                len(cluster.members),
            )
        object_names: list[str] = []
        for member_index, member in enumerate(cluster.members):
            if top_k is not None and member_index >= top_k:
                break
            object_names.append(member.object_name)
        if not object_names:
            raise ValueError(f"Cluster {cluster_id} resolved to zero object names for top_k={top_k}.")
        return tuple(object_names)

    def _assets_for_ordered_object_names(
        self,
        object_names: tuple[str, ...],
        *,
        scale_min: int,
        scale_max: int,
    ) -> tuple[SplitObjectAsset, ...]:
        resolved_assets: list[SplitObjectAsset] = []
        for object_name in object_names:
            resolved_assets.extend(
                self._assets_for_object_and_scale_range(
                    object_name,
                    scale_min=scale_min,
                    scale_max=scale_max,
                )
            )
        return tuple(resolved_assets)

    @staticmethod
    def _parse_scale_range(object_set_id: str, *, scale_min: str, scale_max: str) -> tuple[int, int]:
        parsed_min = int(scale_min)
        parsed_max = int(scale_max)
        if parsed_min > parsed_max:
            raise ValueError(
                f"Invalid scale range in object_set_id {object_set_id!r}: {parsed_min} > {parsed_max}."
            )
        return parsed_min, parsed_max

    def resolve_selector(self, selector: str) -> tuple[SplitObjectAsset, ...]:
        normalized = selector.strip()
        if not normalized:
            raise ValueError("selector must not be empty.")

        match = re.fullmatch(
            r"cluster(?P<cluster_id>ALL|\d+)_topk(?P<top_k>ALL|\d+)@s"
            r"(?P<scale_min>\d{3})_(?P<scale_max>\d{3})",
            normalized,
        )
        if match is not None:
            cluster_token = match.group("cluster_id")
            top_k_token = match.group("top_k")
            scale_min, scale_max = self._parse_scale_range(
                selector,
                scale_min=match.group("scale_min"),
                scale_max=match.group("scale_max"),
            )
            selected_cluster_ids = self.all_cluster_ids() if cluster_token == "ALL" else (int(cluster_token),)
            top_k = None if top_k_token == "ALL" else int(top_k_token)
            object_names: list[str] = []
            for cluster_id in selected_cluster_ids:
                object_names.extend(self._cluster_member_object_names(cluster_id, top_k=top_k))
            return self._assets_for_ordered_object_names(tuple(object_names), scale_min=scale_min, scale_max=scale_max)

        match = re.fullmatch(
            r"object:(?P<object_name>.+)@s(?P<scale_min>\d{3})_(?P<scale_max>\d{3})",
            normalized,
        )
        if match is not None:
            scale_min, scale_max = self._parse_scale_range(
                selector,
                scale_min=match.group("scale_min"),
                scale_max=match.group("scale_max"),
            )
            return self._assets_for_object_and_scale_range(
                match.group("object_name"),
                scale_min=scale_min,
                scale_max=scale_max,
            )

        match = re.fullmatch(r"object_scale:(?P<object_scale_key>.+)", normalized)
        if match is not None:
            return (self.asset_by_object_scale_key(match.group("object_scale_key")),)

        raise ValueError(f"Unsupported selector in object_set_id. {OBJECT_SET_ID_HELP} Got: {selector!r}")


def resolve_object_set_query(object_set_id: str) -> tuple[SplitSide, str]:
    normalized = object_set_id.strip()
    if not normalized:
        raise ValueError("object_set_id must not be empty.")
    match = re.fullmatch(
        rf"(?P<split_side>{SPLIT_SIDE_PATTERN})@(?P<selector_kind>cluster|object|object_scale)@(?P<value>.+)",
        normalized,
    )
    if match is None:
        raise ValueError(f"Unsupported object_set_id format. {OBJECT_SET_ID_HELP} Got: {object_set_id!r}")
    selector_kind = match.group("selector_kind")
    value = match.group("value")
    if selector_kind == "cluster":
        selector = value
    elif selector_kind == "object":
        selector = f"object:{value}"
    else:
        selector = f"object_scale:{value}"
    return match.group("split_side"), selector


def _build_cluster_members(raw_members: list[dict[str, object]]) -> tuple[ClusterMemberMeta, ...]:
    members: list[ClusterMemberMeta] = []
    for raw_member in raw_members:
        members.append(
            ClusterMemberMeta(
                object_name=str(raw_member["object_name"]),
                cluster_scale_tag=str(raw_member["cluster_scale_tag"]),
                distance_to_center=float(raw_member["cluster_distance"]),
                distance_to_global_center=float(raw_member["distance_to_global_center"]),
                rank_in_cluster=int(raw_member["cluster_rank"]),
                num_scales=int(raw_member["num_scales"]),
                scale_tags=tuple(str(scale_tag) for scale_tag in raw_member["scale_tags"]),
            )
        )
    return tuple(members)


def load_rl_split_catalog(split_dir: str | Path, *, side: SplitSide = "train") -> RLSplitCatalog:
    resolved_split_dir = Path(split_dir).expanduser().resolve()
    if not resolved_split_dir.exists():
        raise FileNotFoundError(f"RL split directory does not exist: {resolved_split_dir}")

    cluster_path = resolved_split_dir / f"{side}_cluster.json"
    if not cluster_path.exists():
        raise FileNotFoundError(f"RL split cluster file does not exist: {cluster_path}")

    with cluster_path.open("r", encoding="utf-8") as f:
        raw_clusters = json.load(f)

    ordered_assets: list[SplitObjectAsset] = []
    clusters: dict[int, ClusterIndex] = {}
    for raw_cluster_id, raw_cluster in raw_clusters["clusters"].items():
        cluster_id = int(raw_cluster_id)
        for member_index, raw_member in enumerate(raw_cluster["members"]):
            ordered_scales = sorted(raw_member["scales"], key=lambda scale_item: float(scale_item["scale"]))
            for raw_scale in ordered_scales:
                ordered_assets.append(
                    SplitObjectAsset(
                        object_scale_key=str(raw_scale["object_scale_key"]),
                        object_name=str(raw_member["object_name"]),
                        asset_path=str(raw_scale["asset_path"]),
                        mjcf_path=str(raw_scale["mjcf_path"]),
                        usd_path=None if raw_scale["usd_path"] is None else str(raw_scale["usd_path"]),
                        scale_tag=str(raw_scale["scale_tag"]),
                        scale=float(raw_scale["scale"]),
                        cluster_scale_tag=str(raw_member["cluster_scale_tag"]),
                        cluster_id=int(raw_member["cluster_id"]),
                        cluster_distance=float(raw_member["cluster_distance"]),
                        cluster_rank=int(raw_member["cluster_rank"]),
                        distance_to_global_center=float(raw_member["distance_to_global_center"]),
                        cluster_member_index=member_index,
                    )
                )

        clusters[cluster_id] = ClusterIndex(
            cluster_id=cluster_id,
            split_center_object_name=str(raw_cluster["split_center_object_name"]),
            distance_to_global_center=float(raw_cluster["distance_to_global_center"]),
            member_count=int(raw_cluster["member_count"]),
            total_scale_count=int(raw_cluster["total_scale_count"]),
            members=_build_cluster_members(raw_cluster["members"]),
        )

    return RLSplitCatalog(
        split_dir=resolved_split_dir,
        side=side,
        assets=tuple(ordered_assets),
        clusters=clusters,
    )


def resolve_object_selection(query_cfg: ObjectSetQueryCfg) -> ObjectSelection:
    split_side, catalog_object_set_id = resolve_object_set_query(query_cfg.object_set_id)
    catalog = load_rl_split_catalog(query_cfg.rl_split_dir, side=split_side)
    assets = catalog.resolve_selector(catalog_object_set_id)
    if len(assets) == 0:
        raise ValueError(f"Object set {query_cfg.object_set_id!r} resolved to zero assets.")
    return ObjectSelection(
        split_side=split_side,
        object_set_id=query_cfg.object_set_id,
        active_assets=assets,
    )
