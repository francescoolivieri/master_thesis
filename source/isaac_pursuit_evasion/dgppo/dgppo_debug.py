from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

ScalarTracker = Callable[[str, float], None]


@dataclasses.dataclass(kw_only=True)
class DGPPODebugConfig:
    """Runtime diagnostics for the skrl DG-PPO integration."""

    enabled: bool = True
    step_interval: int = 10
    update_interval: int = 1
    minibatch_interval: int = 1
    sample_env_count: int = 8
    log_minibatches: bool = True
    log_jsonl: bool = True
    log_tensorboard_scalars: bool = True
    log_on_done: bool = True
    rnn_done_norm_epsilon: float = 1e-6
    anomaly_abs_threshold: float = 1e6

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> DGPPODebugConfig:
        if data is None:
            return cls()
        defaults = dataclasses.asdict(cls())
        merged = {**defaults, **dict(data)}
        return cls(**{key: merged[key] for key in defaults})


class DGPPOTrainingDiagnostics:
    """Collect focused live-training diagnostics without feeding them back into learning."""

    def __init__(
        self,
        *,
        cfg: DGPPODebugConfig,
        experiment_dir: str | Path,
        track_data: ScalarTracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.track_data = track_data
        self.enabled = bool(cfg.enabled)
        self._rollout_terminated: list[torch.Tensor] = []
        self._rollout_truncated: list[torch.Tensor] = []
        self._rollout_done_reasons: list[torch.Tensor] = []
        self._rollout_episode_lengths: list[torch.Tensor] = []
        self._rollout_rnn_done_norms: list[torch.Tensor] = []
        self._last_update_id = 0
        self._started_at = time.time()

        self.output_dir = Path(experiment_dir) / "dgppo_debug"
        self.jsonl_path = self.output_dir / "events.jsonl"
        if self.enabled and self.cfg.log_jsonl:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_setup(self, *, agent: Any, trainer_cfg: Any | None) -> None:
        if not self.enabled:
            return
        base_env = agent.env.unwrapped if hasattr(agent.env, "unwrapped") else agent.env
        layout = getattr(base_env, "graph_obs_layout", {})
        data = {
            "device": str(agent.device),
            "experiment_dir": str(getattr(agent, "experiment_dir", "")),
            "trainer_cfg": _jsonable(trainer_cfg),
            "num_envs": int(agent.env.num_envs),
            "num_agents": int(agent.env.num_agents),
            "action_shape": tuple(int(v) for v in getattr(agent.env.action_space, "shape", ())),
            "observation_space": str(getattr(agent, "observation_space", "")),
            "action_space": str(getattr(agent, "action_space", "")),
            "n_constraints": int(getattr(base_env, "n_constraints", 0)),
            "cost_components": _jsonable(getattr(base_env, "cost_components", ())),
            "graph_obs_layout": _jsonable(layout),
            "split": {
                "det_envs": int(agent._det_env_ids.numel()) if agent._det_env_ids is not None else 0,
                "stc_envs": int(agent._stoch_env_ids.numel()) if agent._stoch_env_ids is not None else 0,
                "det_first": _first_values(agent._det_env_ids),
                "stc_first": _first_values(agent._stoch_env_ids),
            },
            "hyperparameters": {
                "rollouts": int(agent.rollouts),
                "learning_epochs": int(agent.learning_epochs),
                "mini_batches": int(agent.mini_batches),
                "rnn_step": int(agent.rnn_step),
                "gamma": float(agent.gamma),
                "gae_lambda": float(agent.gae_lambda),
                "bootstrap_on_truncated": bool(getattr(agent, "bootstrap_on_truncated", False)),
                "clip_eps": float(agent.clip_eps),
                "alpha": float(agent.alpha),
                "cbf_eps": float(agent.cbf_eps),
                "cbf_weight": float(agent.cbf_weight),
                "obs_radius": float(agent.obs_radius),
                "lr_policy": float(agent.lr_policy),
                "lr_vl": float(agent.lr_vl),
                "lr_vh": float(agent.lr_vh),
            },
            "rnn": {
                "policy": getattr(agent.policy, "rnn", None) is not None,
                "Vl": getattr(agent.Vl, "rnn", None) is not None,
                "Vh": getattr(agent.Vh, "rnn", None) is not None,
                "state_shape": _shape(agent._policy_rnn_state),
                "vl_state_shape": _shape(agent._vl_rnn_state),
            },
            "diagnostics": dataclasses.asdict(self.cfg),
        }
        self._write("setup", data)

    def record_transition(
        self,
        *,
        timestep: int,
        timesteps: int,
        env: Any,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        agent_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs_state: torch.Tensor,
        log_prob: torch.Tensor,
        value_l: torch.Tensor,
        value_h: torch.Tensor,
        costs: torch.Tensor,
        old_policy_rnn_state: torch.Tensor | None,
        new_policy_rnn_state: torch.Tensor | None,
        vl_rnn_state: torch.Tensor | None,
        stoch_env_ids: torch.Tensor,
        det_env_ids: torch.Tensor,
        memory_cursor: int,
        rollout_length: int,
    ) -> None:
        if not self.enabled:
            return

        terminated_1d = _bool_1d(terminated)
        truncated_1d = _bool_1d(truncated)
        done_1d = terminated_1d | truncated_1d
        done_reason = _done_reasons(env, device=done_1d.device)
        episode_length = _episode_lengths(env, device=done_1d.device)
        self._rollout_terminated.append(terminated_1d.detach().to("cpu"))
        self._rollout_truncated.append(truncated_1d.detach().to("cpu"))
        if done_reason is not None:
            self._rollout_done_reasons.append(done_reason.detach().to("cpu"))
        if episode_length is not None:
            self._rollout_episode_lengths.append(episode_length.detach().to("cpu"))

        rnn_info = self._rnn_transition_info(
            old_state=old_policy_rnn_state,
            new_state=new_policy_rnn_state,
            done=done_1d,
            n_envs=int(env.num_envs),
            n_agents=int(env.num_agents),
        )
        if rnn_info.get("done_new_norms") is not None:
            self._rollout_rnn_done_norms.append(rnn_info["done_new_norms"].detach().to("cpu"))

        anomaly_reasons = self._transition_anomalies(
            tensors={
                "observations": observations,
                "next_observations": next_observations,
                "actions": actions,
                "rewards": rewards,
                "log_prob": log_prob,
                "value_l": value_l,
                "value_h": value_h,
                "costs": costs,
                "policy_rnn": new_policy_rnn_state,
                "vl_rnn": vl_rnn_state,
            },
            done=done_1d,
            rnn_info=rnn_info,
        )
        should_log = self._should_log_step(timestep=timestep, done=done_1d, anomaly_reasons=anomaly_reasons)
        if not should_log:
            return

        stc = stoch_env_ids.detach().long()
        det = det_env_ids.detach().long()
        pos_error = torch.linalg.vector_norm(agent_state[..., :3] - goal_state[..., :3], dim=-1).squeeze(-1)
        speed = torch.linalg.vector_norm(agent_state[..., 3:6], dim=-1).squeeze(-1)
        action_norm = torch.linalg.vector_norm(actions.reshape(int(env.num_envs), int(env.num_agents), -1), dim=-1)
        action_norm = action_norm.squeeze(-1)
        actions_env = actions.reshape(int(env.num_envs), int(env.num_agents), -1)

        data: dict[str, Any] = {
            "timestep": int(timestep),
            "timesteps": int(timesteps),
            "memory_cursor": int(memory_cursor),
            "rollout_length": int(rollout_length),
            "seconds_since_debug_start": float(time.time() - self._started_at),
            "anomaly_reasons": anomaly_reasons,
            "done": {
                "terminated": _count_rate(terminated_1d),
                "truncated": _count_rate(truncated_1d),
                "any": _count_rate(done_1d),
                "by_reason": _reason_counts(done_reason),
                "stc_any": _count_rate(done_1d[stc]),
                "det_any": _count_rate(done_1d[det]),
            },
            "split": {
                "stc_reward": _stats(rewards.reshape(-1)[stc]),
                "det_reward": _stats(rewards.reshape(-1)[det]),
                "stc_pos_error": _stats(pos_error.reshape(-1)[stc]),
                "det_pos_error": _stats(pos_error.reshape(-1)[det]),
                "stc_action_norm": _stats(action_norm.reshape(-1)[stc]),
                "det_action_norm": _stats(action_norm.reshape(-1)[det]),
                "stc_log_prob": _stats(log_prob.reshape(int(env.num_envs), int(env.num_agents))[stc]),
                "det_log_prob": _stats(log_prob.reshape(int(env.num_envs), int(env.num_agents))[det]),
            },
            "tensors": {
                "observations": _stats(observations),
                "next_observations": _stats(next_observations),
                "actions": _stats(actions),
                "actions_by_dim": _per_dim_stats(actions_env),
                "rewards": _stats(rewards),
                "agent_pos": _stats(agent_state[..., :3]),
                "agent_vel": _stats(agent_state[..., 3:6]),
                "goal_pos": _stats(goal_state[..., :3]),
                "obs_state": _stats(obs_state),
                "pos_error": _stats(pos_error),
                "speed": _stats(speed),
                "action_norm": _stats(action_norm),
                "log_prob": _stats(log_prob),
                "value_l": _stats(value_l),
                "value_h": _stats(value_h),
                "costs": _stats(costs),
            },
            "rnn": _jsonable(rnn_info) | {
                "vl": _rnn_global_stats(vl_rnn_state),
                "skrl_reference_reset_required": bool(done_1d.any().item()),
            },
            "env": {
                "episode_length_buf": _stats(episode_length),
                "reward_components": _reward_component_stats(env),
                "observation_consistency": _observation_consistency(env, agent_state, goal_state, obs_state),
                "infos_keys": sorted(str(k) for k in infos.keys()) if isinstance(infos, Mapping) else str(type(infos)),
            },
            "samples": self._sample_envs(
                observations=observations,
                next_observations=next_observations,
                rewards=rewards,
                done=done_1d,
                done_reason=done_reason,
                pos_error=pos_error,
                action_norm=action_norm,
                log_prob=log_prob.reshape(int(env.num_envs), int(env.num_agents)),
            ),
        }
        self._track_transition_scalars(data)
        self._write("transition", data)

    def log_bootstrap(
        self,
        *,
        timestep: int,
        vl_boot: torch.Tensor,
        vh_boot: torch.Tensor,
        vh_boot_reference: torch.Tensor | None,
        policy_bootstrap_rnn_state: torch.Tensor | None,
        policy_reference_rnn_state: torch.Tensor | None,
    ) -> None:
        if not self.enabled:
            return
        data = {
            "timestep": int(timestep),
            "vl_boot": _stats(vl_boot),
            "vh_boot_current": _stats(vh_boot),
            "vh_boot_reference_style": _stats(vh_boot_reference),
            "vh_boot_abs_diff": _stats(
                None if vh_boot_reference is None else torch.abs(vh_boot.detach() - vh_boot_reference.detach())
            ),
            "policy_bootstrap_rnn": _rnn_global_stats(policy_bootstrap_rnn_state),
            "policy_reference_rnn": _rnn_global_stats(policy_reference_rnn_state),
        }
        if vh_boot_reference is not None:
            diff = torch.abs(vh_boot.detach() - vh_boot_reference.detach())
            self._track("DGPPO Debug/bootstrap_vh_abs_diff_mean", _finite_mean(diff))
            self._track("DGPPO Debug/bootstrap_vh_abs_diff_max", _finite_max(diff))
        self._write("bootstrap", data)

    def log_update_start(
        self,
        *,
        timestep: int,
        update_id: int,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
        ql: torch.Tensor,
        qh_det: torch.Tensor,
        adv_info: dict[str, torch.Tensor],
        cbf_scale: float,
        chunk_ids: torch.Tensor,
    ) -> None:
        if not self.enabled or not self._should_log_update(update_id):
            return
        terminated, truncated, reasons, episode_lengths = self._rollout_boundary_tensors()
        done = None if terminated is None else (terminated | truncated)
        data = {
            "timestep": int(timestep),
            "update_id": int(update_id),
            "cbf_scale": float(cbf_scale),
            "chunk_ids": _jsonable(chunk_ids),
            "rollout_boundaries": {
                "terminated": _stats(terminated),
                "truncated": _stats(truncated),
                "done": _stats(done),
                "done_by_reason": _reason_counts(reasons),
                "episode_length_buf": _stats(episode_lengths),
                "rnn_done_new_norms": _stats(_cat_or_none(self._rollout_rnn_done_norms)),
                # DGPPO DEBUG FIX START: update diagnostics for consumed rollout masks.
                "mask_status": "DGPPO memory/update consumes terminated, truncated, and done boundary masks.",
                # DGPPO DEBUG FIX END: update diagnostics for consumed rollout masks.
            },
            "stochastic_rollout": {
                "reward": _stats(-view["bT_l"]),
                "cost_l": _stats(view["bT_l"]),
                "costs_h": _stats(view["bTah_hs"]),
                # DGPPO DEBUG FIX START: expose stored stochastic masks in debug events.
                "terminated": _stats(view.get("bT_terminated")),
                "truncated": _stats(view.get("bT_truncated")),
                # DGPPO DEBUG FIX END: expose stored stochastic masks in debug events.
                "value_l": _stats(view["bT_Vl"]),
                "value_l_tp1": _stats(view["bTp1_Vl"]),
                "value_h": _stats(view["bTah_Vh"]),
                "value_h_tp1": _stats(view["bTp1ah_Vh"]),
                "old_logp": _stats(view["bTa_logp"]),
                "actions": _stats(view["bTa_actions"]),
            },
            "deterministic_rollout": {
                "reward": _stats(-det_view["bT_l"]),
                "cost_l": _stats(det_view["bT_l"]),
                "costs_h": _stats(det_view["bTah_hs"]),
                # DGPPO DEBUG FIX START: expose stored deterministic masks in debug events.
                "terminated": _stats(det_view.get("bT_terminated")),
                "truncated": _stats(det_view.get("bT_truncated")),
                # DGPPO DEBUG FIX END: expose stored deterministic masks in debug events.
                "value_l": _stats(det_view["bT_Vl"]),
                "value_h": _stats(det_view["bTah_Vh"]),
                "old_logp": _stats(det_view["bTa_logp"]),
                "actions": _stats(det_view["bTa_actions"]),
            },
            "targets_and_advantages": {
                "ql": _stats(ql),
                "qh_det": _stats(qh_det),
                "vl_error": _stats(ql - view["bT_Vl"]),
                "vh_det_error": _stats(qh_det - det_view["bTah_Vh"]),
                "advantage": _stats(adv_info["bTa_A"]),
                "advantage_positive_rate": _positive_rate(adv_info["bTa_A"]),
                "adv_raw": _stats(adv_info["bT_Al_raw"]),
                "adv_norm": _stats(adv_info["bT_Al_norm"]),
                "reward_adv_agent": _stats(adv_info.get("bTa_Al")),
                "reward_adv_used_rate": _true_rate(adv_info.get("bTa_reward_used")),
                "cbf_deriv": _stats(adv_info["bTah_cbf_deriv"]),
                "cbf_penalty": _stats(adv_info["bTah_Acbf"]),
                "cbf_penalty_agent": _stats(adv_info.get("bTa_cbf_penalty")),
                "cbf_active_rate": _true_rate(adv_info.get("bTa_cbf_active")),
                "advantage_before_flip": _stats(adv_info.get("bTa_A_before_flip")),
                "safe_rate": _finite_mean(adv_info["bTa_is_safe"].float()),
            },
        }
        self._track("DGPPO Debug/rollout_done_rate", _finite_mean(done.float()) if done is not None else math.nan)
        truncated_rate = _finite_mean(truncated.float()) if truncated is not None else math.nan
        self._track("DGPPO Debug/rollout_truncated_rate", truncated_rate)
        self._track("DGPPO Debug/advantage_mean", _finite_mean(adv_info["bTa_A"]))
        self._track("DGPPO Debug/advantage_abs_max", _finite_abs_max(adv_info["bTa_A"]))
        self._track("DGPPO Debug/cbf_deriv_max", _finite_max(adv_info["bTah_cbf_deriv"]))
        self._track("DGPPO Debug/cbf_active_rate", _true_rate(adv_info.get("bTa_cbf_active")))
        self._write("update_start", data)

    def log_minibatch(
        self,
        *,
        timestep: int,
        update_id: int,
        epoch: int,
        minibatch: int,
        idx: torch.Tensor,
        info: dict[str, torch.Tensor],
    ) -> None:
        if not self.enabled or not self.cfg.log_minibatches:
            return
        if self.cfg.minibatch_interval > 1 and update_id % self.cfg.minibatch_interval:
            return
        data = {
            "timestep": int(timestep),
            "update_id": int(update_id),
            "epoch": int(epoch),
            "minibatch": int(minibatch),
            "env_indices": _first_values(idx, limit=32),
            "loss_policy": _scalar(info.get("loss_p")),
            "loss_vl": _scalar(info.get("loss_vl")),
            "loss_vh": _scalar(info.get("loss_vh")),
            "clip_frac": _scalar(info.get("clip_frac")),
            "ratio": _stats(info.get("ratio")),
            "log_prob": _stats(info.get("log_prob")),
            "old_logp": _stats(info.get("old_logp")),
            "log_prob_delta": _stats(info.get("log_prob_delta")),
            "log_prob_delta_abs": _stats(None if info.get("log_prob_delta") is None else info["log_prob_delta"].abs()),
            "log_prob_delta_abs_counts": _threshold_counts(info.get("log_prob_delta"), thresholds=(1.0, 5.0, 10.0)),
            "advantages": _stats(info.get("advantages")),
            "entropy": _stats(info.get("entropy")),
            "policy_grad_norm": _scalar(info.get("policy_grad_norm")),
            "vl_grad_norm": _scalar(info.get("vl_grad_norm")),
            "vh_grad_norm": _scalar(info.get("vh_grad_norm")),
            "vl_prediction": _stats(info.get("vl")),
            "vh_prediction": _stats(info.get("vh")),
        }
        delta = info.get("log_prob_delta")
        if isinstance(delta, torch.Tensor):
            self._track("DGPPO Debug/minibatch_log_prob_delta_abs_max", _finite_abs_max(delta))
        ratio = info.get("ratio")
        if isinstance(ratio, torch.Tensor):
            self._track("DGPPO Debug/minibatch_ratio_max", _finite_max(ratio))
        self._write("minibatch", data)

    def log_update_end(
        self,
        *,
        timestep: int,
        update_id: int,
        summary: dict[str, float],
    ) -> None:
        if not self.enabled:
            return
        self._last_update_id = int(update_id)
        self._write("update_end", {"timestep": int(timestep), "update_id": int(update_id), "summary": summary})

    def reset_rollout(self) -> None:
        self._rollout_terminated.clear()
        self._rollout_truncated.clear()
        self._rollout_done_reasons.clear()
        self._rollout_episode_lengths.clear()
        self._rollout_rnn_done_norms.clear()

    def _should_log_step(self, *, timestep: int, done: torch.Tensor, anomaly_reasons: list[str]) -> bool:
        interval = max(1, int(self.cfg.step_interval))
        if timestep % interval == 0:
            return True
        if anomaly_reasons:
            return True
        return bool(self.cfg.log_on_done and done.any().item())

    def _should_log_update(self, update_id: int) -> bool:
        interval = max(1, int(self.cfg.update_interval))
        return update_id % interval == 0

    def _transition_anomalies(
        self,
        *,
        tensors: Mapping[str, torch.Tensor | None],
        done: torch.Tensor,
        rnn_info: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        threshold = float(self.cfg.anomaly_abs_threshold)
        for name, tensor in tensors.items():
            if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            finite = torch.isfinite(tensor)
            if not bool(finite.all().item()):
                reasons.append(f"{name}:non_finite")
                continue
            if threshold > 0 and float(torch.max(torch.abs(tensor)).item()) > threshold:
                reasons.append(f"{name}:abs_gt_{threshold:g}")
        if done.any().item() and float(rnn_info.get("done_new_nonzero_rate", 0.0)) > 0.0:
            reasons.append("rnn_state_nonzero_after_done")
        return reasons

    def _rnn_transition_info(
        self,
        *,
        old_state: torch.Tensor | None,
        new_state: torch.Tensor | None,
        done: torch.Tensor,
        n_envs: int,
        n_agents: int,
    ) -> dict[str, Any]:
        old_norms = _rnn_env_norms(old_state, n_envs=n_envs, n_agents=n_agents)
        new_norms = _rnn_env_norms(new_state, n_envs=n_envs, n_agents=n_agents)
        delta_norms = None
        if old_state is not None and new_state is not None and old_state.shape == new_state.shape:
            delta_norms = _rnn_env_norms(new_state - old_state, n_envs=n_envs, n_agents=n_agents)
        done_new_norms = new_norms[done] if new_norms is not None else None
        eps = float(self.cfg.rnn_done_norm_epsilon)
        done_new_nonzero_rate = 0.0
        done_new_nonzero_count = 0
        if done_new_norms is not None and done_new_norms.numel() > 0:
            nonzero = done_new_norms > eps
            done_new_nonzero_count = int(nonzero.sum().item())
            done_new_nonzero_rate = float(nonzero.float().mean().item())
        return {
            "old": _stats(old_norms),
            "new": _stats(new_norms),
            "delta": _stats(delta_norms),
            "done_new_norms": done_new_norms,
            "done_new_norm": _stats(done_new_norms),
            "done_new_nonzero_count": done_new_nonzero_count,
            "done_new_nonzero_rate": done_new_nonzero_rate,
            "done_reset_expected_by_skrl_ppo_rnn": bool(done.any().item()),
        }

    def _sample_envs(
        self,
        *,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        done_reason: torch.Tensor | None,
        pos_error: torch.Tensor,
        action_norm: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> list[dict[str, Any]]:
        sample_count = max(0, int(self.cfg.sample_env_count))
        if sample_count <= 0:
            return []
        n_envs = int(done.shape[0])
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)
        base_ids = torch.arange(min(sample_count, n_envs), device=done.device)
        ids = torch.unique(torch.cat([done_ids[:sample_count], base_ids]))[:sample_count]
        samples = []
        rewards_1d = rewards.reshape(-1)
        for env_id_tensor in ids:
            env_id = int(env_id_tensor.item())
            item = {
                "env_id": env_id,
                "reward": float(rewards_1d[env_id].detach().item()),
                "done": bool(done[env_id].detach().item()),
                "done_reason": None if done_reason is None else int(done_reason[env_id].detach().item()),
                "pos_error": float(pos_error.reshape(-1)[env_id].detach().item()),
                "action_norm": float(action_norm.reshape(-1)[env_id].detach().item()),
                "log_prob": _jsonable(log_prob[env_id]),
                "obs": _first_values(observations[env_id], limit=16),
                "next_obs": _first_values(next_observations[env_id], limit=16),
            }
            samples.append(item)
        return samples

    def _rollout_boundary_tensors(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        terminated = _stack_or_none(self._rollout_terminated)
        truncated = _stack_or_none(self._rollout_truncated)
        reasons = _stack_or_none(self._rollout_done_reasons)
        episode_lengths = _stack_or_none(self._rollout_episode_lengths)
        return terminated, truncated, reasons, episode_lengths

    def _track_transition_scalars(self, data: Mapping[str, Any]) -> None:
        if not self.cfg.log_tensorboard_scalars:
            return
        done = data["done"]
        tensors = data["tensors"]
        rnn = data["rnn"]
        split = data["split"]
        self._track("DGPPO Debug/transition_done_rate", done["any"]["rate"])
        self._track("DGPPO Debug/transition_terminated_rate", done["terminated"]["rate"])
        self._track("DGPPO Debug/transition_truncated_rate", done["truncated"]["rate"])
        self._track("DGPPO Debug/reward_mean", tensors["rewards"]["mean"])
        self._track("DGPPO Debug/pos_error_mean", tensors["pos_error"]["mean"])
        self._track("DGPPO Debug/action_norm_mean", tensors["action_norm"]["mean"])
        self._track("DGPPO Debug/log_prob_mean", tensors["log_prob"]["mean"])
        self._track("DGPPO Debug/value_l_mean", tensors["value_l"]["mean"])
        self._track("DGPPO Debug/value_h_mean", tensors["value_h"]["mean"])
        self._track("DGPPO Debug/costs_max", tensors["costs"]["max"])
        self._track("DGPPO Debug/stc_reward_mean", split["stc_reward"]["mean"])
        self._track("DGPPO Debug/det_reward_mean", split["det_reward"]["mean"])
        self._track("DGPPO Debug/rnn_done_new_nonzero_rate", rnn["done_new_nonzero_rate"])

    def _track(self, tag: str, value: float | int | None) -> None:
        if self.track_data is None or value is None:
            return
        try:
            value_f = float(value)
        except Exception:
            return
        if math.isfinite(value_f):
            self.track_data(tag, value_f)

    def _write(self, event: str, data: Mapping[str, Any]) -> None:
        if not self.enabled or not self.cfg.log_jsonl:
            return
        payload = {
            "event": event,
            "wall_time": time.time(),
            "data": _jsonable(data),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _stats(tensor: Any) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.as_tensor(tensor)
        except Exception:
            return {"present": False, "repr": repr(tensor)}
    data = tensor.detach()
    result: dict[str, Any] = {
        "present": True,
        "shape": tuple(int(v) for v in data.shape),
        "dtype": str(data.dtype),
        "numel": int(data.numel()),
    }
    if data.numel() == 0:
        return result | {"empty": True}
    if data.dtype == torch.bool:
        values = data.float()
        result.update({
            "true_count": int(data.sum().item()),
            "true_rate": float(values.mean().item()),
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        })
        return result
    if not torch.is_floating_point(data) and not torch.is_complex(data):
        data = data.to(torch.float32)
    if torch.is_complex(data):
        data = torch.abs(data)
    finite = torch.isfinite(data)
    result["finite_count"] = int(finite.sum().item())
    result["finite_rate"] = float(finite.float().mean().item())
    if not bool(finite.any().item()):
        return result
    finite_data = data[finite].to(torch.float32)
    result.update({
        "mean": float(finite_data.mean().item()),
        "std": float(finite_data.std(unbiased=False).item()) if finite_data.numel() > 1 else 0.0,
        "min": float(finite_data.min().item()),
        "max": float(finite_data.max().item()),
        "abs_max": float(torch.max(torch.abs(finite_data)).item()),
    })
    return result


def _count_rate(mask: torch.Tensor) -> dict[str, float | int]:
    mask = _bool_1d(mask)
    count = int(mask.sum().item())
    total = int(mask.numel())
    return {"count": count, "total": total, "rate": float(count / max(total, 1))}


def _bool_1d(tensor: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tensor, dtype=torch.bool, device=tensor.device).reshape(-1)


def _done_reasons(env: Any, *, device: torch.device) -> torch.Tensor | None:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base_env, "get_last_episode_status"):
        get_status = base_env.get_last_episode_status
    elif hasattr(base_env, "get_last_done_reasons"):
        get_status = base_env.get_last_done_reasons
    else:
        return None
    try:
        reasons = get_status()
    except Exception:
        return None
    return torch.as_tensor(reasons, device=device).reshape(-1)


def _episode_lengths(env: Any, *, device: torch.device) -> torch.Tensor | None:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    value = getattr(base_env, "episode_length_buf", None)
    if value is None:
        return None
    return torch.as_tensor(value, device=device).reshape(-1)


def _reward_component_stats(env: Any) -> dict[str, Any]:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if not hasattr(base_env, "get_last_reward_components"):
        return {}
    try:
        components = base_env.get_last_reward_components()
    except Exception:
        return {}
    if not isinstance(components, Mapping):
        return {}
    return {str(key): _stats(value) for key, value in components.items()}


def _observation_consistency(
    env: Any,
    agent_state: torch.Tensor,
    goal_state: torch.Tensor,
    obs_state: torch.Tensor,
) -> dict[str, Any]:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    result: dict[str, Any] = {}
    try:
        robot = getattr(base_env, "_robot")
        terrain = getattr(base_env, "_terrain", None)
        env_origins = getattr(terrain, "env_origins", None)
        root_pos = robot.data.root_pos_w
        if env_origins is not None:
            root_pos = root_pos - env_origins
        root_vel = robot.data.root_lin_vel_w
        obs_pos = agent_state[..., :3].reshape(root_pos.shape)
        obs_vel = agent_state[..., 3:6].reshape(root_vel.shape)
        result["agent_pos_abs_diff"] = _stats(torch.abs(obs_pos - root_pos))
        result["agent_vel_abs_diff"] = _stats(torch.abs(obs_vel - root_vel))
    except Exception as exc:
        result["agent_state_check_error"] = repr(exc)

    try:
        if hasattr(base_env, "get_reference_pose"):
            ref_pos, _ref_yaw = base_env.get_reference_pose()
            obs_goal = goal_state[..., :3].reshape(ref_pos.shape)
            result["goal_pos_abs_diff"] = _stats(torch.abs(obs_goal - ref_pos))
    except Exception as exc:
        result["goal_state_check_error"] = repr(exc)

    try:
        cfg = getattr(base_env, "cfg", None)
        obstacle_mode = getattr(cfg, "obstacle_observation_mode", "pillars")
        pillar_xy = getattr(base_env, "_pillar_positions_xy", None)
        if obstacle_mode == "pillars" and pillar_xy is not None and obs_state.shape[1] > 0:
            obs_xy = obs_state[..., :2]
            expected = pillar_xy.unsqueeze(0).expand(obs_xy.shape[0], -1, -1)
            result["obstacle_xy_abs_diff"] = _stats(torch.abs(obs_xy - expected))
        elif obstacle_mode == "ray_caster" and obs_state.shape[1] > 0:
            obs_xy = obs_state[..., :2]
            result["ray_obstacle_xy"] = _stats(obs_xy)
            result["ray_obstacle_distance"] = _stats(
                torch.linalg.vector_norm(obs_xy - agent_state[:, :1, :2], dim=-1)
            )
    except Exception as exc:
        result["obstacle_state_check_error"] = repr(exc)
    return result


def _reason_counts(reasons: torch.Tensor | None) -> dict[str, int]:
    if reasons is None or reasons.numel() == 0:
        return {}
    flat = reasons.detach().reshape(-1).to("cpu")
    unique, counts = torch.unique(flat, return_counts=True)
    return {str(int(key.item())): int(value.item()) for key, value in zip(unique, counts)}


def _rnn_env_norms(rnn_state: torch.Tensor | None, *, n_envs: int, n_agents: int) -> torch.Tensor | None:
    if rnn_state is None or not isinstance(rnn_state, torch.Tensor) or rnn_state.numel() == 0:
        return None
    if rnn_state.ndim != 4:
        return None
    layers, total_agents, carries, hidden = rnn_state.shape
    expected = n_envs * n_agents
    if int(total_agents) != int(expected):
        return None
    reshaped = rnn_state.detach().reshape(layers, n_envs, n_agents, carries, hidden)
    return torch.linalg.vector_norm(reshaped.float(), dim=(0, 2, 3, 4))


def _rnn_global_stats(rnn_state: torch.Tensor | None) -> dict[str, Any]:
    norm = None if rnn_state is None else torch.linalg.vector_norm(rnn_state.float())
    return {"state": _stats(rnn_state), "norm": _stats(norm)}


def _stack_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    return torch.stack(values, dim=0)


def _cat_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    non_empty = [value.reshape(-1) for value in values if value.numel() > 0]
    if not non_empty:
        return None
    return torch.cat(non_empty, dim=0)


def _first_values(tensor: Any, *, limit: int = 8) -> list[Any]:
    if tensor is None:
        return []
    if isinstance(tensor, torch.Tensor):
        values = tensor.detach().reshape(-1)[:limit].to("cpu").tolist()
    else:
        try:
            values = list(tensor)[:limit]
        except Exception:
            return [repr(tensor)]
    return [_jsonable(value) for value in values]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _scalar(value)
        return _first_values(value, limit=64)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    with suppress(Exception):
        if isinstance(value, torch.dtype):
            return str(value)
    return repr(value)


def _shape(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    if tensor is None:
        return None
    return tuple(int(v) for v in tensor.shape)


def _scalar(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        item = value.detach().reshape(-1)[0].item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, (float, int, bool)):
        return value
    try:
        return float(value)
    except Exception:
        return None


def _finite_mean(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return math.nan
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return math.nan
    return float(values[finite].mean().item())


def _finite_max(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return math.nan
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return math.nan
    return float(values[finite].max().item())


def _finite_abs_max(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return math.nan
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return math.nan
    return float(values[finite].abs().max().item())


def _positive_rate(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return math.nan
    return float((tensor.detach() > 0).float().mean().item())


def _true_rate(tensor: Any) -> float:
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return math.nan
    return float(tensor.detach().bool().float().mean().item())


def _threshold_counts(tensor: Any, *, thresholds: tuple[float, ...]) -> dict[str, Any]:
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return {"present": False}
    values = tensor.detach().abs().float()
    finite = values[torch.isfinite(values)]
    result: dict[str, Any] = {"present": True, "numel": int(values.numel()), "finite_count": int(finite.numel())}
    if finite.numel() == 0:
        return result
    for threshold in thresholds:
        mask = finite > float(threshold)
        result[f"abs_gt_{threshold:g}_count"] = int(mask.sum().item())
        result[f"abs_gt_{threshold:g}_rate"] = float(mask.float().mean().item())
    return result


def _per_dim_stats(tensor: Any) -> dict[str, Any]:
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return {"present": False}
    data = tensor.detach()
    if data.ndim == 0:
        return {"present": False, "reason": "scalar"}
    flat = data.reshape(-1, data.shape[-1]).float()
    finite = torch.isfinite(flat)
    safe = torch.where(finite, flat, torch.zeros_like(flat))
    counts = finite.sum(dim=0).clamp_min(1)
    means = safe.sum(dim=0) / counts
    mins = torch.where(finite, flat, torch.full_like(flat, float("inf"))).min(dim=0).values
    maxs = torch.where(finite, flat, torch.full_like(flat, float("-inf"))).max(dim=0).values
    centered = torch.where(finite, flat - means, torch.zeros_like(flat))
    stds = torch.sqrt((centered.square().sum(dim=0) / counts).clamp_min(0.0))
    return {
        "present": True,
        "shape": tuple(int(v) for v in data.shape),
        "mean": _jsonable(means),
        "std": _jsonable(stds),
        "min": _jsonable(mins),
        "max": _jsonable(maxs),
        "finite_count": _jsonable(finite.sum(dim=0)),
    }
