"""Shared utilities for moving-task twist distillation."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


@dataclass
class TwistMiniBatch:
    observations: TensorDict
    student_pbvs_twist_b: torch.Tensor
    teacher_desired_twist_b: torch.Tensor
    return_phase_mask: torch.Tensor
    teacher_joint_command_target: torch.Tensor
    student_joint_release: torch.Tensor
    student_joint_home_delta: torch.Tensor
    student_joint_current_pos: torch.Tensor
    joint_label_mask: torch.Tensor
    dones: torch.Tensor


class SuccessfulTrajectoryDataset:
    """Flat transition dataset assembled from successful full-episode trajectories."""

    _TRAIN_KEYS = (
        "student_pbvs_twist_b",
        "teacher_desired_twist_b",
        "return_phase_mask",
        "teacher_joint_command_target",
        "student_joint_release",
        "student_joint_home_delta",
        "student_joint_current_pos",
        "joint_label_mask",
        "dones",
    )
    _SAVE_KEYS = (
        "episode_ids",
        "condition_ids",
        "time_indices",
    )

    def __init__(
        self,
        sample_device: str,
        shard_dir: str | None = None,
        shards_per_mini_batch: int = 8,
        max_replay_episodes: int = 0,
        max_replay_samples: int = 0,
        anchor_replay_fraction: float = 0.0,
        max_anchor_shards: int = 0,
        current_replay_fraction: float = 0.0,
        current_replay_shards: int = 1,
    ) -> None:
        self.sample_device = sample_device
        self.shard_dir = shard_dir
        self.shards_per_mini_batch = max(int(shards_per_mini_batch), 1)
        self.max_replay_episodes = max(int(max_replay_episodes), 0)
        self.max_replay_samples = max(int(max_replay_samples), 0)
        self.anchor_replay_fraction = min(max(float(anchor_replay_fraction), 0.0), 1.0)
        self.max_anchor_shards = max(int(max_anchor_shards), 0)
        self.current_replay_fraction = min(max(float(current_replay_fraction), 0.0), 1.0)
        self.current_replay_shards = max(int(current_replay_shards), 1)
        self.observation_chunks: list[torch.Tensor] = []
        self.chunks: dict[str, list[torch.Tensor]] = {key: [] for key in (*self._TRAIN_KEYS, *self._SAVE_KEYS)}
        self.shards: list[dict[str, object]] = []
        self._next_shard_index = 0
        self._num_samples = 0
        self._num_episodes = 0
        self._frozen_observations: torch.Tensor | None = None
        self._frozen: dict[str, torch.Tensor] | None = None
        if self.shard_dir is not None:
            os.makedirs(self.shard_dir, exist_ok=True)

    @property
    def filled_steps(self) -> int:
        return self._num_samples

    def num_samples(self) -> int:
        return self._num_samples

    def num_episodes(self) -> int:
        return self._num_episodes

    def load_shard_index(self, path: str) -> None:
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if "shards" not in payload:
            raise ValueError(f"Dataset file is not a shard-backed successful-trajectory index: {path}")
        self.shards = list(payload["shards"])
        self._num_samples = int(payload.get("num_samples", sum(int(shard["num_samples"]) for shard in self.shards)))
        self._num_episodes = int(
            payload.get("num_episodes", sum(int(shard["num_episodes"]) for shard in self.shards))
        )
        self._next_shard_index = len(self.shards)
        self.observation_chunks.clear()
        for value in self.chunks.values():
            value.clear()
        self._frozen_observations = None
        self._frozen = None

    def add_successful_trajectories(
        self,
        step_samples: list[dict[str, torch.Tensor]],
        selected_env_mask: torch.Tensor,
        *,
        condition_ids: torch.Tensor,
    ) -> int:
        condition_ids_cpu = condition_ids.detach().cpu().long()
        selected_env_mask_cpu = selected_env_mask.detach().cpu().bool() & (condition_ids_cpu >= 0)
        num_selected_envs = int(selected_env_mask_cpu.sum().item())
        if num_selected_envs <= 0:
            return 0

        selected_condition_ids = condition_ids_cpu[selected_env_mask_cpu].long()
        selected_episode_ids = torch.arange(
            self._num_episodes,
            self._num_episodes + num_selected_envs,
            dtype=torch.long,
        )
        shard_observation_chunks: list[torch.Tensor] = []
        shard_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in (*self._TRAIN_KEYS, *self._SAVE_KEYS)}
        for time_idx, sample in enumerate(step_samples):
            shard_observation_chunks.append(sample["observations_student"][selected_env_mask_cpu].cpu())
            for key in self._TRAIN_KEYS:
                shard_chunks[key].append(sample[key][selected_env_mask_cpu].cpu())
            shard_chunks["episode_ids"].append(selected_episode_ids.clone())
            shard_chunks["condition_ids"].append(selected_condition_ids.clone())
            shard_chunks["time_indices"].append(torch.full((num_selected_envs,), time_idx, dtype=torch.long))

        shard_observations = torch.cat(shard_observation_chunks, dim=0)
        shard_tensors = {key: torch.cat(value, dim=0) for key, value in shard_chunks.items() if value}
        num_samples = int(shard_observations.shape[0])
        if self.shard_dir is None:
            self.observation_chunks.append(shard_observations)
            for key, value in shard_tensors.items():
                self.chunks[key].append(value)
        else:
            shard_index = self._next_shard_index
            self._next_shard_index += 1
            shard_path = os.path.join(self.shard_dir, f"shard_{shard_index:06d}.pt")
            torch.save(
                {
                    "observations": {"student": shard_observations},
                    **shard_tensors,
                    "num_samples": num_samples,
                    "num_episodes": num_selected_envs,
                },
                shard_path,
            )
            self.shards.append(
                {
                    "path": shard_path,
                    "num_samples": num_samples,
                    "num_episodes": num_selected_envs,
                    "is_anchor": False,
                }
            )
            self._num_samples += num_samples
            self._num_episodes += num_selected_envs
            self._evict_old_shards_if_needed()
            self._frozen_observations = None
            self._frozen = None
            return num_selected_envs

        self._num_samples += num_samples
        self._num_episodes += num_selected_envs
        self._frozen_observations = None
        self._frozen = None
        return num_selected_envs

    def add_transition_steps(
        self,
        step_samples: list[dict[str, torch.Tensor]],
        valid_env_masks: list[torch.Tensor],
        *,
        condition_ids: torch.Tensor,
    ) -> int:
        if len(step_samples) != len(valid_env_masks):
            raise ValueError(
                f"step_samples and valid_env_masks must have the same length, got "
                f"{len(step_samples)} and {len(valid_env_masks)}."
            )
        if not step_samples:
            return 0

        condition_ids_cpu = condition_ids.detach().cpu().long()
        valid_masks_cpu = [mask.detach().cpu().bool() & (condition_ids_cpu >= 0) for mask in valid_env_masks]
        episode_mask = torch.stack(valid_masks_cpu, dim=0).any(dim=0)
        num_episodes = int(episode_mask.sum().item())
        if num_episodes <= 0:
            return 0

        episode_ids_by_env = torch.full_like(condition_ids_cpu, -1)
        episode_ids_by_env[episode_mask] = torch.arange(
            self._num_episodes,
            self._num_episodes + num_episodes,
            dtype=torch.long,
        )

        shard_observation_chunks: list[torch.Tensor] = []
        shard_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in (*self._TRAIN_KEYS, *self._SAVE_KEYS)}
        for time_idx, (sample, valid_mask_cpu) in enumerate(zip(step_samples, valid_masks_cpu, strict=True)):
            if not torch.any(valid_mask_cpu):
                continue
            shard_observation_chunks.append(sample["observations_student"][valid_mask_cpu].cpu())
            for key in self._TRAIN_KEYS:
                shard_chunks[key].append(sample[key][valid_mask_cpu].cpu())
            shard_chunks["episode_ids"].append(episode_ids_by_env[valid_mask_cpu].clone())
            shard_chunks["condition_ids"].append(condition_ids_cpu[valid_mask_cpu].clone())
            shard_chunks["time_indices"].append(torch.full((int(valid_mask_cpu.sum().item()),), time_idx, dtype=torch.long))

        if not shard_observation_chunks:
            return 0

        shard_observations = torch.cat(shard_observation_chunks, dim=0)
        shard_tensors = {key: torch.cat(value, dim=0) for key, value in shard_chunks.items() if value}
        num_samples = int(shard_observations.shape[0])
        if self.shard_dir is None:
            self.observation_chunks.append(shard_observations)
            for key, value in shard_tensors.items():
                self.chunks[key].append(value)
        else:
            shard_index = self._next_shard_index
            self._next_shard_index += 1
            shard_path = os.path.join(self.shard_dir, f"shard_{shard_index:06d}.pt")
            torch.save(
                {
                    "observations": {"student": shard_observations},
                    **shard_tensors,
                    "num_samples": num_samples,
                    "num_episodes": num_episodes,
                },
                shard_path,
            )
            self.shards.append(
                {
                    "path": shard_path,
                    "num_samples": num_samples,
                    "num_episodes": num_episodes,
                    "is_anchor": False,
                }
            )
            self._num_samples += num_samples
            self._num_episodes += num_episodes
            self._evict_old_shards_if_needed()
            self._frozen_observations = None
            self._frozen = None
            return num_episodes

        self._num_samples += num_samples
        self._num_episodes += num_episodes
        self._frozen_observations = None
        self._frozen = None
        return num_episodes

    def _evict_old_shards_if_needed(self) -> None:
        if self.shard_dir is None:
            return
        while self.shards and (
            (self.max_replay_episodes > 0 and self._num_episodes > self.max_replay_episodes)
            or (self.max_replay_samples > 0 and self._num_samples > self.max_replay_samples)
        ):
            evict_index = next(
                (index for index, shard in enumerate(self.shards) if not bool(shard.get("is_anchor", False))),
                None,
            )
            if evict_index is None:
                break
            shard = self.shards.pop(evict_index)
            self._num_samples -= int(shard["num_samples"])
            self._num_episodes -= int(shard["num_episodes"])
            try:
                os.remove(str(shard["path"]))
            except FileNotFoundError:
                pass

    def promote_latest_shard_to_anchor(self, *, metadata: dict[str, object] | None = None) -> bool:
        if not self.shards:
            return False
        shard = self.shards[-1]
        if bool(shard.get("is_anchor", False)):
            return False
        shard["is_anchor"] = True
        if metadata:
            shard["anchor_metadata"] = dict(metadata)
        self._trim_anchor_shards_if_needed()
        return True

    def num_anchor_shards(self) -> int:
        return sum(1 for shard in self.shards if bool(shard.get("is_anchor", False)))

    def drop_latest_non_anchor_shard(self) -> bool:
        if not self.shards:
            return False
        shard = self.shards[-1]
        if bool(shard.get("is_anchor", False)):
            return False
        self.shards.pop()
        self._num_samples -= int(shard["num_samples"])
        self._num_episodes -= int(shard["num_episodes"])
        try:
            os.remove(str(shard["path"]))
        except FileNotFoundError:
            pass
        self._frozen_observations = None
        self._frozen = None
        return True

    def _trim_anchor_shards_if_needed(self) -> None:
        if self.max_anchor_shards <= 0:
            return
        anchor_indices = [index for index, shard in enumerate(self.shards) if bool(shard.get("is_anchor", False))]
        while len(anchor_indices) > self.max_anchor_shards:
            demote_index = anchor_indices.pop(0)
            self.shards[demote_index]["is_anchor"] = False
            self.shards[demote_index].pop("anchor_metadata", None)

    def freeze(self) -> None:
        if self._frozen is not None:
            return
        if not self.observation_chunks:
            raise RuntimeError("Cannot freeze an empty successful-trajectory dataset.")
        self._frozen_observations = torch.cat(self.observation_chunks, dim=0)
        self.observation_chunks.clear()
        self._frozen = {}
        for key, value in self.chunks.items():
            if not value:
                continue
            self._frozen[key] = torch.cat(value, dim=0)
            value.clear()

    def random_mini_batch_generator(
        self,
        *,
        num_batches: int,
        mini_batch_size: int,
        episodes_per_iteration: int = 0,
    ) -> Iterator[TwistMiniBatch]:
        if self.shards:
            yield from self._random_mini_batch_generator_from_shards(
                num_batches=num_batches,
                mini_batch_size=mini_batch_size,
                episodes_per_iteration=episodes_per_iteration,
            )
            return
        self.freeze()
        assert self._frozen is not None
        assert self._frozen_observations is not None
        dataset_size = int(self._frozen_observations.shape[0])
        sample_pool = torch.arange(dataset_size)
        if episodes_per_iteration > 0:
            episode_ids = self._frozen["episode_ids"]
            unique_episode_ids = torch.unique(episode_ids)
            take = min(int(episodes_per_iteration), int(unique_episode_ids.numel()))
            chosen_episode_ids = unique_episode_ids[torch.randperm(unique_episode_ids.numel())[:take]]
            sample_pool = torch.nonzero(torch.isin(episode_ids, chosen_episode_ids), as_tuple=False).flatten()
            if sample_pool.numel() <= 0:
                raise RuntimeError("Episode-subset sampling produced an empty transition pool.")
        for _ in range(num_batches):
            local_indices = torch.randint(low=0, high=int(sample_pool.numel()), size=(mini_batch_size,))
            indices = sample_pool[local_indices]
            observations = TensorDict(
                {"student": self._frozen_observations[indices].to(self.sample_device)},
                batch_size=[mini_batch_size],
                device=self.sample_device,
            )
            yield TwistMiniBatch(
                observations=observations,
                student_pbvs_twist_b=self._frozen["student_pbvs_twist_b"][indices].to(self.sample_device),
                teacher_desired_twist_b=self._frozen["teacher_desired_twist_b"][indices].to(self.sample_device),
                return_phase_mask=self._frozen["return_phase_mask"][indices].to(self.sample_device),
                teacher_joint_command_target=self._frozen["teacher_joint_command_target"][indices].to(self.sample_device),
                student_joint_release=self._frozen["student_joint_release"][indices].to(self.sample_device),
                student_joint_home_delta=self._frozen["student_joint_home_delta"][indices].to(self.sample_device),
                student_joint_current_pos=self._frozen["student_joint_current_pos"][indices].to(self.sample_device),
                joint_label_mask=self._frozen["joint_label_mask"][indices].to(self.sample_device),
                dones=self._frozen["dones"][indices].to(self.sample_device),
            )

    def _random_mini_batch_generator_from_shards(
        self,
        *,
        num_batches: int,
        mini_batch_size: int,
        episodes_per_iteration: int = 0,
    ) -> Iterator[TwistMiniBatch]:
        if not self.shards:
            raise RuntimeError("Cannot sample from an empty shard-backed successful-trajectory dataset.")
        eligible_shard_indices = self._sample_eligible_shards(episodes_per_iteration)
        if episodes_per_iteration > 0:
            observations_cpu, batch_cpu = self._build_episode_balanced_transition_pool(
                eligible_shard_indices,
                episodes_per_iteration,
            )
            pool_size = int(observations_cpu.shape[0])
            for _ in range(num_batches):
                indices = torch.randint(low=0, high=pool_size, size=(mini_batch_size,))
                observations = TensorDict(
                    {"student": observations_cpu[indices].to(self.sample_device)},
                    batch_size=[mini_batch_size],
                    device=self.sample_device,
                )
                yield TwistMiniBatch(
                    observations=observations,
                    student_pbvs_twist_b=batch_cpu["student_pbvs_twist_b"][indices].to(self.sample_device),
                    teacher_desired_twist_b=batch_cpu["teacher_desired_twist_b"][indices].to(self.sample_device),
                    return_phase_mask=batch_cpu["return_phase_mask"][indices].to(self.sample_device),
                    teacher_joint_command_target=batch_cpu["teacher_joint_command_target"][indices].to(
                        self.sample_device
                    ),
                    student_joint_release=batch_cpu["student_joint_release"][indices].to(self.sample_device),
                    student_joint_home_delta=batch_cpu["student_joint_home_delta"][indices].to(self.sample_device),
                    student_joint_current_pos=batch_cpu["student_joint_current_pos"][indices].to(self.sample_device),
                    joint_label_mask=batch_cpu["joint_label_mask"][indices].to(self.sample_device),
                    dones=batch_cpu["dones"][indices].to(self.sample_device),
                )
            return

        if self.shard_dir is not None and (
            self.anchor_replay_fraction > 0.0 or self.current_replay_fraction > 0.0
        ):
            anchor_shard_indices, history_shard_indices, current_shard_indices = self._partition_replay_shards(
                eligible_shard_indices
            )

            for _ in range(num_batches):
                observations_cpu, batch_cpu = self._sample_from_replay_pools(
                    mini_batch_size=mini_batch_size,
                    anchor_shard_indices=anchor_shard_indices,
                    history_shard_indices=history_shard_indices,
                    current_shard_indices=current_shard_indices,
                )
                permutation = torch.randperm(mini_batch_size)
                observations = TensorDict(
                    {"student": observations_cpu[permutation].to(self.sample_device)},
                    batch_size=[mini_batch_size],
                    device=self.sample_device,
                )
                yield TwistMiniBatch(
                    observations=observations,
                    student_pbvs_twist_b=batch_cpu["student_pbvs_twist_b"][permutation].to(self.sample_device),
                    teacher_desired_twist_b=batch_cpu["teacher_desired_twist_b"][permutation].to(
                        self.sample_device
                    ),
                    return_phase_mask=batch_cpu["return_phase_mask"][permutation].to(self.sample_device),
                    teacher_joint_command_target=batch_cpu["teacher_joint_command_target"][permutation].to(
                        self.sample_device
                    ),
                    student_joint_release=batch_cpu["student_joint_release"][permutation].to(self.sample_device),
                    student_joint_home_delta=batch_cpu["student_joint_home_delta"][permutation].to(
                        self.sample_device
                    ),
                    student_joint_current_pos=batch_cpu["student_joint_current_pos"][permutation].to(
                        self.sample_device
                    ),
                    joint_label_mask=batch_cpu["joint_label_mask"][permutation].to(self.sample_device),
                    dones=batch_cpu["dones"][permutation].to(self.sample_device),
                )
            return

        shard_weights = self._sample_weights_for_shards(eligible_shard_indices, episodes_per_iteration)
        for _ in range(num_batches):
            observations_cpu, batch_cpu = self._sample_transitions_from_shards(
                eligible_shard_indices,
                mini_batch_size,
                shard_weights=shard_weights,
            )
            observations = TensorDict(
                {"student": observations_cpu.to(self.sample_device)},
                batch_size=[mini_batch_size],
                device=self.sample_device,
            )
            yield TwistMiniBatch(
                observations=observations,
                student_pbvs_twist_b=batch_cpu["student_pbvs_twist_b"].to(self.sample_device),
                teacher_desired_twist_b=batch_cpu["teacher_desired_twist_b"].to(self.sample_device),
                return_phase_mask=batch_cpu["return_phase_mask"].to(self.sample_device),
                teacher_joint_command_target=batch_cpu["teacher_joint_command_target"].to(self.sample_device),
                student_joint_release=batch_cpu["student_joint_release"].to(self.sample_device),
                student_joint_home_delta=batch_cpu["student_joint_home_delta"].to(self.sample_device),
                student_joint_current_pos=batch_cpu["student_joint_current_pos"].to(self.sample_device),
                joint_label_mask=batch_cpu["joint_label_mask"].to(self.sample_device),
                dones=batch_cpu["dones"].to(self.sample_device),
            )

    def _sample_transitions_from_shards(
        self,
        shard_indices: list[int],
        mini_batch_size: int,
        *,
        shard_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}.")
        if not shard_indices:
            raise RuntimeError("Cannot sample transitions from an empty shard list.")
        if shard_weights is None:
            shard_weights = torch.tensor(
                [int(self.shards[index]["num_samples"]) for index in shard_indices],
                dtype=torch.float32,
            )
        if torch.sum(shard_weights) <= 0:
            raise RuntimeError("Cannot sample transitions from shards with zero total weight.")

        num_active_shards = min(self.shards_per_mini_batch, len(shard_indices), mini_batch_size)
        local_active_indices = torch.multinomial(
            shard_weights,
            num_samples=num_active_shards,
            replacement=False,
        )
        active_shard_indices = [shard_indices[int(index.item())] for index in local_active_indices]
        active_weights = shard_weights[local_active_indices]
        sample_counts = torch.multinomial(
            active_weights / active_weights.sum(),
            num_samples=mini_batch_size,
            replacement=True,
        ).bincount(minlength=len(active_shard_indices))

        observation_parts: list[torch.Tensor] = []
        batch_parts: dict[str, list[torch.Tensor]] = {key: [] for key in self._TRAIN_KEYS}
        for local_shard_idx, count_tensor in enumerate(sample_counts):
            count = int(count_tensor.item())
            if count <= 0:
                continue
            shard_index = int(active_shard_indices[local_shard_idx])
            shard = torch.load(str(self.shards[shard_index]["path"]), weights_only=False, map_location="cpu")
            shard_size = int(shard["num_samples"])
            indices = torch.randint(low=0, high=shard_size, size=(count,))
            observation_parts.append(shard["observations"]["student"][indices])
            for key in self._TRAIN_KEYS:
                batch_parts[key].append(shard[key][indices])

        observations_cpu = torch.cat(observation_parts, dim=0)
        batch_cpu = {key: torch.cat(value, dim=0) for key, value in batch_parts.items()}
        return observations_cpu, batch_cpu

    def _partition_replay_shards(self, shard_indices: list[int]) -> tuple[list[int], list[int], list[int]]:
        anchor_shard_indices = [
            index for index in shard_indices if bool(self.shards[index].get("is_anchor", False))
        ]
        non_anchor_indices = [
            index for index in shard_indices if not bool(self.shards[index].get("is_anchor", False))
        ]
        current_take = min(self.current_replay_shards, len(non_anchor_indices))
        current_shard_indices = non_anchor_indices[-current_take:] if current_take > 0 else []
        history_shard_indices = non_anchor_indices[:-current_take] if current_take > 0 else non_anchor_indices
        return anchor_shard_indices, history_shard_indices, current_shard_indices

    def _sample_from_replay_pools(
        self,
        *,
        mini_batch_size: int,
        anchor_shard_indices: list[int],
        history_shard_indices: list[int],
        current_shard_indices: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pool_specs = [
            ("anchor", anchor_shard_indices, self.anchor_replay_fraction),
            ("current", current_shard_indices, self.current_replay_fraction),
            (
                "history",
                history_shard_indices,
                max(1.0 - self.anchor_replay_fraction - self.current_replay_fraction, 0.0),
            ),
        ]
        available_specs = [(name, indices, fraction) for name, indices, fraction in pool_specs if indices]
        if not available_specs:
            raise RuntimeError("Cannot sample from replay pools: all pools are empty.")

        counts = {name: 0 for name, _, _ in available_specs}
        remaining = mini_batch_size
        for name, indices, fraction in available_specs:
            if fraction <= 0.0:
                continue
            count = int(round(float(mini_batch_size) * fraction))
            if count <= 0 and mini_batch_size >= len(available_specs):
                count = 1
            count = min(count, remaining)
            counts[name] = count
            remaining -= count

        while remaining > 0:
            target_name = "history" if any(name == "history" for name, _, _ in available_specs) else available_specs[0][0]
            counts[target_name] += remaining
            remaining = 0

        observation_parts: list[torch.Tensor] = []
        batch_parts: dict[str, list[torch.Tensor]] = {key: [] for key in self._TRAIN_KEYS}
        for name, shard_indices, _fraction in available_specs:
            count = counts[name]
            if count <= 0:
                continue
            observations_cpu, batch_cpu = self._sample_transitions_from_shards(shard_indices, count)
            observation_parts.append(observations_cpu)
            for key in self._TRAIN_KEYS:
                batch_parts[key].append(batch_cpu[key])

        observations = torch.cat(observation_parts, dim=0)
        batch = {key: torch.cat(value, dim=0) for key, value in batch_parts.items()}
        if int(observations.shape[0]) != mini_batch_size:
            raise RuntimeError(
                f"Replay pool sampler returned {int(observations.shape[0])} samples, expected {mini_batch_size}."
            )
        return observations, batch

    def _build_episode_balanced_transition_pool(
        self,
        shard_indices: list[int],
        episodes_per_iteration: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if episodes_per_iteration <= 0:
            raise ValueError("episodes_per_iteration must be positive for balanced transition pool sampling.")
        if not shard_indices:
            raise RuntimeError("Cannot build a transition pool from an empty shard list.")
        base_quota = episodes_per_iteration // len(shard_indices)
        remainder = episodes_per_iteration % len(shard_indices)
        observation_parts: list[torch.Tensor] = []
        batch_parts: dict[str, list[torch.Tensor]] = {key: [] for key in self._TRAIN_KEYS}
        for local_idx, shard_index in enumerate(shard_indices):
            quota = base_quota + (1 if local_idx < remainder else 0)
            if quota <= 0:
                continue
            shard = torch.load(str(self.shards[shard_index]["path"]), weights_only=False, map_location="cpu")
            episode_ids = torch.unique(shard["episode_ids"])
            take = min(int(quota), int(episode_ids.numel()))
            if take <= 0:
                continue
            chosen_episode_ids = episode_ids[torch.randperm(int(episode_ids.numel()))[:take]]
            selected_mask = torch.isin(shard["episode_ids"], chosen_episode_ids)
            if not torch.any(selected_mask):
                continue
            observation_parts.append(shard["observations"]["student"][selected_mask])
            for key in self._TRAIN_KEYS:
                batch_parts[key].append(shard[key][selected_mask])

        if not observation_parts:
            raise RuntimeError("Episode-balanced shard sampling produced an empty transition pool.")
        observations_cpu = torch.cat(observation_parts, dim=0)
        batch_cpu = {key: torch.cat(value, dim=0) for key, value in batch_parts.items()}
        return observations_cpu, batch_cpu

    def _sample_eligible_shards(self, episodes_per_iteration: int) -> list[int]:
        if episodes_per_iteration > 0:
            # Use every shard for episode-subset training. Each shard typically
            # corresponds to a contiguous bucket-condition block, so selecting
            # only one or two whole shards creates large loss jumps between
            # iterations. The per-shard weights below approximate
            # episodes_per_iteration / num_shards episodes from each shard.
            return list(range(len(self.shards)))
        return list(range(len(self.shards)))

    def _sample_weights_for_shards(self, shard_indices: list[int], episodes_per_iteration: int) -> torch.Tensor:
        if episodes_per_iteration <= 0:
            return torch.tensor(
                [int(self.shards[index]["num_samples"]) for index in shard_indices],
                dtype=torch.float32,
            )
        if not shard_indices:
            raise RuntimeError("Cannot compute shard weights for an empty shard list.")
        base_quota = episodes_per_iteration // len(shard_indices)
        remainder = episodes_per_iteration % len(shard_indices)
        weights: list[float] = []
        for local_idx, shard_index in enumerate(shard_indices):
            quota = base_quota + (1 if local_idx < remainder else 0)
            num_episodes = int(self.shards[shard_index]["num_episodes"])
            num_samples = int(self.shards[shard_index]["num_samples"])
            if num_episodes <= 0 or num_samples <= 0:
                weights.append(0.0)
                continue
            selected_episodes = max(min(quota, num_episodes), 1)
            samples_per_episode = num_samples / float(num_episodes)
            weights.append(float(selected_episodes) * samples_per_episode)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if torch.sum(weight_tensor) <= 0:
            raise RuntimeError("Episode-balanced shard sampling produced zero total weight.")
        return weight_tensor

    def save(self, path: str, *, metadata: dict | None = None) -> None:
        if self.shards:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(
                {
                    "metadata": metadata or {},
                    "shards": self.shards,
                    "num_samples": self._num_samples,
                    "num_episodes": self._num_episodes,
                },
                path,
            )
            return
        self.freeze()
        assert self._frozen is not None
        assert self._frozen_observations is not None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "metadata": metadata or {},
                "observations": {"student": self._frozen_observations},
                **self._frozen,
            },
            path,
        )


class TwistDistillationBase:
    """Common model, label, loss, and env stepping utilities for twist distillation."""

    def __init__(self, env, train_cfg: dict, log_dir: str, device: str = "cpu") -> None:
        self.env = env
        self.train_cfg = copy.deepcopy(train_cfg)
        self.log_dir = log_dir
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.save_interval = int(self.train_cfg.get("save_interval", 100))
        self.current_learning_iteration = 0
        clip_actions = self.train_cfg.get("clip_actions", None)
        self.clip_actions = None if clip_actions is None else float(clip_actions)

        obs = self.env.get_observations().to(self.device)
        alg_cfg = self.train_cfg.get("algorithm", {})
        self.teacher_policy_obs_group = str(alg_cfg.get("teacher_policy_obs_group", "teacher"))
        obs_group_cfg = copy.deepcopy(self.train_cfg["obs_groups"])
        obs_group_cfg["teacher"] = [self.teacher_policy_obs_group]
        self.train_cfg["obs_groups"] = obs_group_cfg
        self.obs_groups = resolve_obs_groups(obs, obs_group_cfg, ["student", "teacher"])
        self.student, self.teacher = self._build_models(obs)
        self.teacher_loaded = False

        optimizer_name = alg_cfg.get("optimizer", "adam")
        learning_rate = float(alg_cfg.get("learning_rate", 3.0e-4))
        self.optimizer = resolve_optimizer(optimizer_name)(self.student.parameters(), lr=learning_rate)
        self.gradient_length = int(alg_cfg.get("gradient_length", 1))
        self.max_grad_norm = alg_cfg.get("max_grad_norm", 1.0)
        self.root_loss_weight = float(alg_cfg.get("root_twist_loss_weight", 1.0))
        self.root_linear_loss_weight = float(alg_cfg.get("root_linear_twist_loss_weight", 0.5))
        self.root_angular_loss_weight = float(alg_cfg.get("root_angular_twist_loss_weight", 0.5))
        self.joint_command_target_loss_weight = float(
            alg_cfg.get("joint_command_target_loss_weight", alg_cfg.get("joint_action_loss_weight", 1.0))
        )
        self.updates_per_iteration = int(alg_cfg.get("updates_per_iteration", 16))
        self.mini_batch_size = int(alg_cfg.get("mini_batch_size", 32768))
        self.episodes_per_iteration = int(alg_cfg.get("episodes_per_iteration", 0))
        self.shards_per_mini_batch = int(alg_cfg.get("shards_per_mini_batch", 8))
        self.max_replay_episodes = int(alg_cfg.get("max_replay_episodes", 0))
        self.max_replay_samples = int(alg_cfg.get("max_replay_samples", 0))
        self.anchor_replay_fraction = float(alg_cfg.get("anchor_replay_fraction", 0.0))
        self.max_anchor_shards = int(alg_cfg.get("max_anchor_shards", 0))
        self.current_replay_fraction = float(alg_cfg.get("current_replay_fraction", 0.0))
        self.current_replay_shards = int(alg_cfg.get("current_replay_shards", 1))
        self.stochastic_rollout = bool(alg_cfg.get("stochastic_rollout", False))

        teacher_linear_scale = alg_cfg.get("teacher_rl_linear_twist_scale")
        teacher_angular_scale = alg_cfg.get("teacher_rl_angular_twist_scale")
        student_linear_scale = alg_cfg.get("student_rl_linear_twist_scale")
        student_angular_scale = alg_cfg.get("student_rl_angular_twist_scale")
        if teacher_linear_scale is None or teacher_angular_scale is None:
            raise ValueError(
                "Twist distillation requires explicit teacher_rl_linear_twist_scale "
                "and teacher_rl_angular_twist_scale."
            )
        self.teacher_residual_scale = torch.tensor(
            [float(teacher_linear_scale)] * 3 + [float(teacher_angular_scale)] * 3,
            dtype=torch.float32,
            device=self.device,
        )
        print(
            "[INFO] TwistDistillation teacher_label_residual_scale="
            f"linear:{float(teacher_linear_scale)} angular:{float(teacher_angular_scale)}",
            flush=True,
        )
        print(f"[INFO] TwistDistillation clip_actions={self.clip_actions}", flush=True)

        self.root_slice, self.joint_slice = self._resolve_action_slices()
        self.teacher_obs_slices = self._resolve_obs_term_slices(self.teacher_policy_obs_group)
        self.student_obs_slices = self._resolve_obs_term_slices("student")
        self.teacher_last_action_slice = self.teacher_obs_slices["last_action"]
        self.student_last_action_slice = self.student_obs_slices["last_action"]
        self.root_term = self.env.unwrapped.action_manager.get_term("floating_root")
        self.joint_term = self.env.unwrapped.action_manager.get_term("hand_joints")
        if student_linear_scale is None or student_angular_scale is None:
            self.student_residual_scale = self.root_term.residual_scale
        else:
            self.student_residual_scale = torch.tensor(
                [float(student_linear_scale)] * 3 + [float(student_angular_scale)] * 3,
                dtype=torch.float32,
                device=self.device,
            )
        print(
            "[INFO] TwistDistillation student_loss_residual_scale="
            f"linear:{float(self.student_residual_scale[0])} angular:{float(self.student_residual_scale[3])}",
            flush=True,
        )
        if self.joint_term.return_mode != self.joint_term.RETURN_MODE_HOLD:
            raise ValueError(
                "Twist distillation only supports hold return mode for joint distillation. "
                f"Got return_mode={self.joint_term.return_mode!r}."
            )

        self._teacher_prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self._student_prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _build_models(self, obs: TensorDict) -> tuple[MLPModel, MLPModel]:
        student_cfg = copy.deepcopy(self.train_cfg["student"])
        teacher_cfg = copy.deepcopy(self.train_cfg["teacher"])
        student_class: type[MLPModel] = resolve_callable(student_cfg.pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(teacher_cfg.pop("class_name"))  # type: ignore
        student = student_class(obs, self.obs_groups, "student", self.num_actions, **student_cfg).to(self.device)
        teacher = teacher_class(obs, self.obs_groups, "teacher", self.num_actions, **teacher_cfg).to(self.device)
        print(f"Student Model: {student}")
        print(f"Teacher Model: {teacher}")
        return student, teacher

    def _resolve_action_slices(self) -> tuple[slice, slice]:
        action_manager = self.env.unwrapped.action_manager
        start = 0
        root_slice = None
        joint_slice = None
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            end = start + term.action_dim
            if term_name == "floating_root":
                root_slice = slice(start, end)
            elif term_name == "hand_joints":
                joint_slice = slice(start, end)
            start = end
        if root_slice is None:
            raise ValueError("Twist distillation requires an action term named 'floating_root'.")
        if joint_slice is None:
            raise ValueError("Twist distillation requires an action term named 'hand_joints'.")
        return root_slice, joint_slice

    def _resolve_obs_term_slices(self, obs_group: str) -> dict[str, slice]:
        observation_manager = self.env.unwrapped.observation_manager
        term_names = observation_manager.active_terms[obs_group]
        term_dims = observation_manager.group_obs_term_dim[obs_group]
        term_slices = {}
        offset = 0
        for term_name, term_dim in zip(term_names, term_dims, strict=True):
            if len(term_dim) != 1:
                raise ValueError(
                    f"Twist distillation expects 1D concatenated observations, got {obs_group}.{term_name} "
                    f"shape={term_dim}."
                )
            width = int(term_dim[-1])
            term_slices[term_name] = slice(offset, offset + width)
            offset += width
        if "last_action" not in term_slices:
            raise ValueError(f"Observation group {obs_group!r} does not contain a last_action term.")
        if term_slices["last_action"].stop - term_slices["last_action"].start != self.num_actions:
            raise ValueError(
                f"Expected {obs_group}.last_action width {self.num_actions}, got "
                f"{term_slices['last_action'].stop - term_slices['last_action'].start}."
            )
        expected_width = self.env.get_observations()[obs_group].shape[-1]
        if offset != expected_width:
            raise ValueError(
                f"Resolved {obs_group} observation width {offset}, but runtime observation width is {expected_width}."
            )
        layout = ", ".join(f"{name}:{slc.start}-{slc.stop}" for name, slc in term_slices.items())
        print(f"[INFO] TwistDistillation {obs_group}_obs_layout={layout}", flush=True)
        return term_slices

    def _obs_with_action_history(
        self,
        obs: TensorDict,
        *,
        teacher_action_history: torch.Tensor,
        student_action_history: torch.Tensor,
    ) -> TensorDict:
        teacher_obs = obs[self.teacher_policy_obs_group].clone()
        student_obs = obs["student"].clone()
        teacher_obs[:, self.teacher_last_action_slice] = teacher_action_history
        student_obs[:, self.student_last_action_slice] = student_action_history
        return TensorDict(
            {self.teacher_policy_obs_group: teacher_obs, "student": student_obs},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def _clip_model_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.clip_actions is None:
            return actions
        return torch.clamp(actions, -self.clip_actions, self.clip_actions)

    def training_sample_to_cpu(self, sample: dict[str, torch.Tensor], dones: torch.Tensor) -> dict[str, torch.Tensor]:
        slim_sample = {"observations_student": sample["observations_student"].detach().cpu()}
        for key in SuccessfulTrajectoryDataset._TRAIN_KEYS:
            if key == "dones":
                slim_sample[key] = dones.detach().cpu()
            else:
                slim_sample[key] = sample[key].detach().cpu()
        return slim_sample

    def load_teacher(self, path: str, strict: bool = True) -> None:
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        state_dict = loaded_dict.get("teacher_state_dict") or loaded_dict.get("actor_state_dict")
        if state_dict is None:
            raise KeyError(f"Checkpoint {path} does not contain teacher_state_dict or actor_state_dict.")
        self.teacher.load_state_dict(state_dict, strict=strict)
        self.teacher_loaded = True
        self.teacher.eval()

    def save(self, path: str, infos: dict | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "teacher_state_dict": self.teacher.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos or {},
            },
            path,
        )

    def make_labeled_sample(
        self,
        obs: TensorDict,
        *,
        teacher_action_history: torch.Tensor,
        student_action_history: torch.Tensor,
        update_normalization: bool = True,
    ) -> dict[str, torch.Tensor]:
        obs = obs.to(self.device)
        obs_for_models = self._obs_with_action_history(
            obs,
            teacher_action_history=teacher_action_history,
            student_action_history=student_action_history,
        )
        with torch.no_grad():
            if update_normalization:
                self.student.update_normalization(obs_for_models)
            student_actions = self._clip_model_actions(
                self.student(obs_for_models, stochastic_output=self.stochastic_rollout)
            ).detach()
            teacher_actions = self._clip_model_actions(self.teacher(obs_for_models)).detach()
            teacher_desired_twist_b = self.root_term.compute_desired_twist_b_for_label(
                teacher_actions[:, self.root_slice],
                target_pose_source="full",
                enable_motion_feedforward=True,
                residual_scale=self.teacher_residual_scale,
            ).detach()
            return_phase_mask = self.root_term.compute_return_phase_mask_for_label().detach()
            student_pbvs_twist_b = self.root_term.compute_pbvs_twist_b_for_label(
                target_pose_source="partial",
                return_phase_mask=return_phase_mask,
            ).detach()
            teacher_joint_actions = teacher_actions[:, self.joint_slice]
            teacher_joint_command_target = self.joint_term.compute_grasp_command_target_for_label(
                teacher_joint_actions,
                target_pose_source="full",
            ).detach()
            (
                student_joint_release,
                student_joint_home_delta,
                student_joint_current_pos,
            ) = self.joint_term.compute_grasp_command_target_context_for_label(target_pose_source="partial")
            joint_label_mask = (
                self.joint_term.compute_grasp_command_target_controllable_mask(student_joint_release)
                & ~return_phase_mask
            ).detach()

        return {
            "observations_student": obs_for_models["student"].detach(),
            "teacher_observations": obs_for_models[self.teacher_policy_obs_group].detach(),
            "student_actions": student_actions.detach(),
            "teacher_actions": teacher_actions.detach(),
            "student_pbvs_twist_b": student_pbvs_twist_b.detach(),
            "teacher_desired_twist_b": teacher_desired_twist_b.detach(),
            "return_phase_mask": return_phase_mask.detach(),
            "teacher_joint_command_target": teacher_joint_command_target.detach(),
            "student_joint_release": student_joint_release.detach(),
            "student_joint_home_delta": student_joint_home_delta.detach(),
            "student_joint_current_pos": student_joint_current_pos.detach(),
            "joint_label_mask": joint_label_mask.detach(),
        }

    def update_from_dataset(self, dataset: SuccessfulTrajectoryDataset) -> dict[str, float]:
        total_root_loss = 0.0
        total_root_linear_loss = 0.0
        total_root_angular_loss = 0.0
        total_joint_loss = 0.0
        total_loss = 0.0
        count = 0
        grad_steps = 0
        accumulated_loss: torch.Tensor | None = None

        for batch in dataset.random_mini_batch_generator(
            num_batches=self.updates_per_iteration,
            mini_batch_size=self.mini_batch_size,
            episodes_per_iteration=self.episodes_per_iteration,
        ):
            student_actions = self.student(batch.observations)
            root_actions = student_actions[:, self.root_slice]
            student_desired_twist_b = torch.clamp(
                batch.student_pbvs_twist_b
                + self.student_residual_scale.unsqueeze(0) * torch.clamp(root_actions, -1.0, 1.0),
                -self.root_term.twist_limit,
                self.root_term.twist_limit,
            )
            grasp_mask = ~batch.return_phase_mask
            if grasp_mask.any():
                root_error = student_desired_twist_b[grasp_mask] - batch.teacher_desired_twist_b[grasp_mask]
                root_linear_norm = torch.clamp(self.root_term.twist_limit[:3].to(root_error.device), min=1.0e-6)
                root_angular_norm = torch.clamp(self.root_term.twist_limit[3:].to(root_error.device), min=1.0e-6)
                root_linear_loss = torch.mean(torch.square(root_error[:, :3] / root_linear_norm.unsqueeze(0)))
                root_angular_loss = torch.mean(torch.square(root_error[:, 3:] / root_angular_norm.unsqueeze(0)))
                root_loss = (
                    self.root_linear_loss_weight * root_linear_loss
                    + self.root_angular_loss_weight * root_angular_loss
                )
            else:
                root_linear_loss = student_desired_twist_b.sum() * 0.0
                root_angular_loss = student_desired_twist_b.sum() * 0.0
                root_loss = root_linear_loss + root_angular_loss

            joint_actions = student_actions[:, self.joint_slice]
            if batch.joint_label_mask.any():
                student_joint_command_target = self.joint_term.compute_grasp_command_target_from_context_for_loss(
                    joint_actions,
                    batch.student_joint_release,
                    batch.student_joint_home_delta,
                    batch.student_joint_current_pos,
                )
                joint_loss = nn.functional.mse_loss(
                    student_joint_command_target[batch.joint_label_mask],
                    batch.teacher_joint_command_target[batch.joint_label_mask],
                )
            else:
                joint_loss = student_actions.sum() * 0.0

            loss = self.root_loss_weight * root_loss + self.joint_command_target_loss_weight * joint_loss
            accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
            count += 1
            total_root_loss += root_loss.item()
            total_root_linear_loss += root_linear_loss.item()
            total_root_angular_loss += root_angular_loss.item()
            total_joint_loss += joint_loss.item()
            total_loss += loss.item()

            if count % self.gradient_length == 0:
                self.optimizer_step(accumulated_loss)
                accumulated_loss = None
                grad_steps += 1
            self.student.reset(batch.dones.view(-1))

        if accumulated_loss is not None:
            self.optimizer_step(accumulated_loss)
            grad_steps += 1

        divisor = max(count, 1)
        return {
            "loss": total_loss / divisor,
            "root_twist": total_root_loss / divisor,
            "root_linear_twist": total_root_linear_loss / divisor,
            "root_angular_twist": total_root_angular_loss / divisor,
            "joint_command_target": total_joint_loss / divisor,
            "grad_steps": float(grad_steps),
            "buffer_steps": float(dataset.filled_steps),
            "buffer_samples": float(dataset.num_samples()),
            "buffer_episodes": float(dataset.num_episodes()),
        }

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), float(self.max_grad_norm))
        self.optimizer.step()
        self.student.detach_hidden_state()

    def manual_step_no_auto_reset(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_env = self.env.unwrapped
        base_env.action_manager.process_action(self._clip_model_actions(actions).to(base_env.device))
        for _ in range(base_env.cfg.decimation):
            base_env._sim_step_counter += 1
            base_env.action_manager.apply_action()
            base_env.scene.write_data_to_sim()
            base_env.sim.step(render=False)
            if base_env.sim.has_gui() or base_env.sim.has_rtx_sensors():
                if base_env._sim_step_counter % base_env.cfg.sim.render_interval == 0:
                    base_env.sim.render()
            base_env.scene.update(dt=base_env.physics_dt)

        base_env.episode_length_buf += 1
        base_env.common_step_counter += 1
        reset_buf = base_env.termination_manager.compute()
        reward = base_env.reward_manager.compute(dt=base_env.step_dt)
        return reset_buf, reward

    def refresh_command_metrics(self) -> None:
        self.env.unwrapped.command_manager.compute(dt=self.env.unwrapped.step_dt)

    def refresh_after_optional_reset(self, reset_env_ids: torch.Tensor | None = None) -> TensorDict:
        base_env = self.env.unwrapped
        if reset_env_ids is not None and len(reset_env_ids) > 0:
            base_env._reset_idx(reset_env_ids)
        base_env.command_manager.compute(dt=base_env.step_dt)
        if "interval" in base_env.event_manager.available_modes:
            base_env.event_manager.apply(mode="interval", dt=base_env.step_dt)
        base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
        return TensorDict(base_env.obs_buf, batch_size=[base_env.num_envs], device=base_env.device)

    def goal_condition_masks(self) -> dict[str, torch.Tensor]:
        goal_term = self.env.unwrapped.command_manager.get_term("goal")
        metrics = goal_term.metrics
        object_at_goal_pos = metrics.get("object_at_goal_pos", None)
        hand_at_goal = metrics.get("hand_at_goal", None)
        contact_gate = metrics.get("grasp_contact_active", None)

        if object_at_goal_pos is None:
            object_at_goal_pos = metrics["object_goal_pos_error"] <= goal_term.cfg.success_pos_threshold
        else:
            object_at_goal_pos = object_at_goal_pos.bool()

        if hand_at_goal is None:
            hand_pos_error = metrics["hand_goal_pos_error"]
            hand_pos_ok = hand_pos_error <= goal_term.cfg.success_pos_threshold
            if hasattr(goal_term.cfg, "use_orientation_in_success") and goal_term.cfg.use_orientation_in_success:
                hand_rot_error = metrics["hand_goal_rot_error"]
                hand_at_goal = hand_pos_ok & (hand_rot_error <= goal_term.cfg.success_rot_threshold)
            else:
                hand_at_goal = hand_pos_ok
        else:
            hand_at_goal = hand_at_goal.bool()

        if contact_gate is None:
            contact_gate = torch.zeros_like(object_at_goal_pos, dtype=torch.bool)
        else:
            contact_gate = contact_gate.bool()

        hand_object_success = object_at_goal_pos & hand_at_goal
        return {
            "hand_success": hand_at_goal,
            "object_pos_success": object_at_goal_pos,
            "contact_gate": contact_gate,
            "hand_object_success": hand_object_success,
            "hand_object_contact_success": hand_object_success & contact_gate,
        }

    def log_iteration(
        self,
        iteration: int,
        loss_dict: dict[str, float],
        mean_reward: float,
        collection_time: float,
        learn_time: float,
    ) -> None:
        self.writer.add_scalar("Loss/root_twist", loss_dict["root_twist"], iteration)
        self.writer.add_scalar("Loss/root_linear_twist", loss_dict["root_linear_twist"], iteration)
        self.writer.add_scalar("Loss/root_angular_twist", loss_dict["root_angular_twist"], iteration)
        self.writer.add_scalar("Loss/joint_command_target", loss_dict["joint_command_target"], iteration)
        self.writer.add_scalar("Loss/total", loss_dict["loss"], iteration)
        self.writer.add_scalar("Buffer/steps", loss_dict["buffer_steps"], iteration)
        self.writer.add_scalar("Buffer/samples", loss_dict["buffer_samples"], iteration)
        self.writer.add_scalar("Buffer/episodes", loss_dict["buffer_episodes"], iteration)
        self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
        self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
        self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        print(
            f"Iteration {iteration}/{self.train_cfg.get('max_iterations', '?')} | "
            f"loss={loss_dict['loss']:.5f} root={loss_dict['root_twist']:.5f} "
            f"root_lin={loss_dict['root_linear_twist']:.5f} root_ang={loss_dict['root_angular_twist']:.5f} "
            f"joint={loss_dict['joint_command_target']:.5f} reward={mean_reward:.3f} "
            f"time={collection_time + learn_time:.2f}s",
            flush=True,
        )
