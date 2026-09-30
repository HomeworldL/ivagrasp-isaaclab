"""Online all-transition DAgger for moving-task twist distillation."""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from .common import SuccessfulTrajectoryDataset, TwistDistillationBase


class TwistDaggerRunner(TwistDistillationBase):
    """Online student rollout with teacher labels over all visited transitions."""

    def __init__(self, env, train_cfg: dict, log_dir: str, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.storage = SuccessfulTrajectoryDataset(
            sample_device=self.device,
            shard_dir=os.path.join(log_dir, "student_dagger_shards"),
            shards_per_mini_batch=self.shards_per_mini_batch,
            max_replay_episodes=self.max_replay_episodes,
            max_replay_samples=self.max_replay_samples,
            anchor_replay_fraction=self.anchor_replay_fraction,
            max_anchor_shards=self.max_anchor_shards,
            current_replay_fraction=self.current_replay_fraction,
            current_replay_shards=self.current_replay_shards,
        )

    def reset_action_histories(self) -> None:
        self._teacher_prev_actions.zero_()
        self._student_prev_actions.zero_()

    def collect_student_rollout_batch(
        self,
        initial_obs: TensorDict,
        *,
        max_steps: int,
    ) -> dict[str, float]:
        if max_steps > int(self.env.unwrapped.max_episode_length):
            raise ValueError(
                f"max_steps={max_steps} exceeds env max_episode_length={int(self.env.unwrapped.max_episode_length)}."
            )

        obs = initial_obs.to(self.device)
        num_envs = self.env.unwrapped.num_envs
        step_samples: list[dict[str, torch.Tensor]] = []
        valid_env_masks: list[torch.Tensor] = []
        excluded_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        total_reward = 0.0

        self.student.train()
        self.teacher.eval()
        for _step_idx in range(max_steps):
            if torch.all(excluded_env_mask):
                break
            sample = self.make_labeled_sample(
                obs,
                teacher_action_history=self._teacher_prev_actions,
                student_action_history=self._student_prev_actions,
                update_normalization=True,
            )
            student_actions = sample["student_actions"].clone()
            student_actions[excluded_env_mask] = 0.0

            valid_env_masks.append((~excluded_env_mask).detach().cpu())
            reset_buf, reward = self.manual_step_no_auto_reset(student_actions)
            active_reward_mask = ~excluded_env_mask
            if torch.any(active_reward_mask):
                total_reward += float(reward[active_reward_mask].mean().item())
            else:
                total_reward += float(reward.mean().item())
            invalid_term = self.env.unwrapped.termination_manager.get_term("invalid_state").bool().to(self.device)
            object_too_far_term = self.env.unwrapped.termination_manager.get_term("object_too_far").bool().to(self.device)
            object_goal_too_far_term = (
                self.env.unwrapped.termination_manager.get_term("object_goal_too_far").bool().to(self.device)
            )
            unresolved_mask = ~excluded_env_mask
            new_excluded_mask = (
                reset_buf.to(self.device, dtype=torch.bool)
                | invalid_term
                | object_too_far_term
                | object_goal_too_far_term
            ) & unresolved_mask

            dones = reset_buf.to(self.device, dtype=torch.bool).detach()
            step_samples.append(self.training_sample_to_cpu(sample, dones))

            excluded_env_mask |= new_excluded_mask
            done_mask = reset_buf.to(device=self.device, dtype=torch.bool).unsqueeze(-1)
            self._teacher_prev_actions[:] = torch.where(
                done_mask,
                torch.zeros_like(self._teacher_prev_actions),
                sample["teacher_actions"].to(self.device),
            )
            self._student_prev_actions[:] = torch.where(
                done_mask,
                torch.zeros_like(self._student_prev_actions),
                sample["student_actions"].to(self.device),
            )
            self._teacher_prev_actions[excluded_env_mask] = 0.0
            self._student_prev_actions[excluded_env_mask] = 0.0
            self.student.reset(reset_buf)
            self.teacher.reset(reset_buf)
            obs = self.refresh_after_optional_reset(None).to(self.device)

        condition_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        added_episodes = self.storage.add_transition_steps(
            step_samples,
            valid_env_masks,
            condition_ids=condition_ids,
        )
        return {
            "mean_reward": total_reward / max(float(len(step_samples)), 1.0),
            "stored_episodes": float(added_episodes),
            "stored_transitions": float(self.storage.num_samples()),
            "stored_total_episodes": float(self.storage.num_episodes()),
        }

    def learn_from_dataset(
        self,
        *,
        num_learning_iterations: int,
        mean_collection_reward: float,
    ) -> dict[str, float]:
        if self.storage.num_samples() <= 0:
            raise RuntimeError("DAgger replay buffer is empty.")
        self.student.train()
        self.teacher.eval()
        last_loss_dict: dict[str, float] = {}
        for _ in range(num_learning_iterations):
            iteration = self.current_learning_iteration
            learn_start = time.time()
            loss_dict = self.update_from_dataset(self.storage)
            last_loss_dict = loss_dict
            learn_time = time.time() - learn_start
            self.log_iteration(iteration, loss_dict, mean_collection_reward, 0.0, learn_time)
            if iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
            self.current_learning_iteration += 1
        return last_loss_dict
