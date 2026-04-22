"""DGPPO training loop for IsaacLab vector environments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .dgppo_losses import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate
from .task_adapters import PosTrackingGraphAdapter


def _extract_policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if len(obs) == 1:
            return next(iter(obs.values()))
        raise KeyError("Expected a 'policy' key in observation dict for DGPPO training.")
    return obs


def _parse_env_step(step_out):
    if len(step_out) != 5:
        raise RuntimeError(f"Unexpected env.step output format for DGPPO: {len(step_out)} values.")
    return step_out


def _parse_env_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...], min_log_std: float, max_log_std: float):
        super().__init__()
        layers = []
        last = obs_dim
        for hid in hidden_sizes:
            layers.append(nn.Linear(last, hid))
            layers.append(nn.ELU())
            last = hid
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(last, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        x = self.backbone(obs)
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class MLPValue(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers = []
        last = obs_dim
        for hid in hidden_sizes:
            layers.append(nn.Linear(last, hid))
            layers.append(nn.ELU())
            last = hid
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class DGPPOTrainStats:
    update: int
    timesteps: int
    loss_total: float
    loss_policy: float
    loss_value_l: float
    loss_value_h: float
    entropy: float
    clip_frac: float
    reward_mean: float
    cbf_unsafe_frac: float


def run_dgppo_training(
    env,
    base_env,
    agent_cfg: dict,
    log_dir: str,
    total_timesteps: int,
    checkpoint_path: str | None = None,
) -> None:
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    obs_space = env.single_observation_space
    if getattr(obs_space, "shape", None) is not None:
        obs_dim = int(obs_space.shape[0])
    elif hasattr(obs_space, "spaces") and "policy" in obs_space.spaces:
        obs_dim = int(obs_space.spaces["policy"].shape[0])
    else:
        raise RuntimeError(f"Unsupported single_observation_space for DGPPO: {obs_space}")
    act_dim = int(env.single_action_space.shape[0])
    num_envs = int(getattr(env, "num_envs", 1))

    models_cfg = agent_cfg.get("models", {})
    policy_model_cfg = models_cfg.get("policy", {})
    value_model_cfg = models_cfg.get("value", {})

    policy_layers = tuple(policy_model_cfg.get("network", [{}])[0].get("layers", [256, 256]))
    value_layers = tuple(value_model_cfg.get("network", [{}])[0].get("layers", [256, 256]))

    algo_cfg = agent_cfg.get("agent", {})
    dgppo_cfg = agent_cfg.get("dgppo", {})

    n_constraint_heads = int(dgppo_cfg.get("n_constraint_heads", 3))
    rollouts = int(algo_cfg.get("rollouts", 16))
    learning_epochs = int(algo_cfg.get("learning_epochs", 4))
    mini_batches = int(algo_cfg.get("mini_batches", 8))
    discount_factor = float(algo_cfg.get("discount_factor", 0.99))
    gae_lambda = float(algo_cfg.get("lambda", 0.95))
    ratio_clip = float(algo_cfg.get("ratio_clip", 0.2))
    entropy_loss_scale = float(algo_cfg.get("entropy_loss_scale", 0.0))
    value_loss_scale = float(algo_cfg.get("value_loss_scale", 1.0))
    learning_rate = float(algo_cfg.get("learning_rate", 3.0e-4))
    grad_norm_clip = float(algo_cfg.get("grad_norm_clip", 0.5))

    alpha = float(dgppo_cfg.get("alpha", 5.0))
    cbf_eps = float(dgppo_cfg.get("cbf_eps", 0.02))
    cbf_weight = float(dgppo_cfg.get("cbf_weight", 1.0))
    cbf_dt = float(dgppo_cfg.get("cbf_dt", getattr(base_env, "step_dt", 0.03)))
    obstacle_clearance = float(dgppo_cfg.get("obstacle_clearance", 0.0))
    value_h_loss_scale = float(dgppo_cfg.get("value_h_loss_scale", 1.0))

    adapter = PosTrackingGraphAdapter(
        obstacle_clearance=obstacle_clearance,
        n_constraint_heads=n_constraint_heads,
    )

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_sizes=policy_layers,
        min_log_std=float(policy_model_cfg.get("min_log_std", -20.0)),
        max_log_std=float(policy_model_cfg.get("max_log_std", 2.0)),
    ).to(device)
    value_l = MLPValue(obs_dim=obs_dim, out_dim=1, hidden_sizes=value_layers).to(device)
    value_h = MLPValue(obs_dim=obs_dim, out_dim=n_constraint_heads, hidden_sizes=value_layers).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_l.parameters()) + list(value_h.parameters()),
        lr=learning_rate,
    )

    checkpoints_dir = Path(log_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        value_l.load_state_dict(checkpoint["value_l"])
        value_h.load_state_dict(checkpoint["value_h"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[INFO] Loaded DGPPO checkpoint: {checkpoint_path}")

    obs = _parse_env_reset(env.reset())
    obs = _extract_policy_obs(obs).to(device)

    timesteps = 0
    update_idx = 0
    updates_total = int(math.ceil(total_timesteps / float(max(1, rollouts))))
    mb_size = max(1, (rollouts * num_envs) // max(1, mini_batches))

    while timesteps < total_timesteps:
        update_idx += 1
        obs_buf = []
        action_buf = []
        logp_buf = []
        reward_buf = []
        done_buf = []
        vl_buf = []
        vh_buf = []
        cost_buf = []

        for _ in range(rollouts):
            with torch.no_grad():
                adapter_data = adapter.extract(base_env)
                dist = policy.dist(obs)
                action_raw = dist.sample()
                action = action_raw.clamp(-1.0, 1.0)
                logp = dist.log_prob(action_raw).sum(dim=-1)
                vl = value_l(obs).squeeze(-1)
                vh = value_h(obs)

            next_obs, reward, terminated, truncated, _ = _parse_env_step(env.step(action))
            reward = reward.to(device)
            if reward.ndim > 1:
                reward = reward.reshape(num_envs, -1).mean(dim=-1)
            terminated = terminated.to(device)
            truncated = truncated.to(device)
            if terminated.ndim > 1:
                terminated = terminated.reshape(num_envs, -1).any(dim=-1)
            if truncated.ndim > 1:
                truncated = truncated.reshape(num_envs, -1).any(dim=-1)
            done = torch.logical_or(terminated.bool(), truncated.bool())

            obs_buf.append(obs)
            action_buf.append(action_raw)
            logp_buf.append(logp)
            reward_buf.append(reward)
            done_buf.append(done.to(device))
            vl_buf.append(vl)
            vh_buf.append(vh)
            cost_buf.append(adapter_data.constraint_costs.squeeze(1).to(device))

            obs = _extract_policy_obs(next_obs).to(device)
            timesteps += num_envs

        with torch.no_grad():
            last_vl = value_l(obs).squeeze(-1)
            last_vh = value_h(obs)

        T_obs = torch.stack(obs_buf, dim=0)  # [T, B, O]
        T_actions = torch.stack(action_buf, dim=0)  # [T, B, A]
        T_logp = torch.stack(logp_buf, dim=0)  # [T, B]
        T_rewards = torch.stack(reward_buf, dim=0)  # [T, B]
        T_vl = torch.stack(vl_buf, dim=0)  # [T, B]
        T_vh = torch.stack(vh_buf, dim=0)  # [T, B, NH]
        T_costs = torch.stack(cost_buf, dim=0)  # [T, B, NH]

        bTah_hs = T_costs.permute(1, 0, 2).unsqueeze(2)  # [B, T, 1, NH]
        bT_l = (-T_rewards).permute(1, 0)  # [B, T]
        bTp1ah_Vh = torch.cat(
            [
                T_vh.permute(1, 0, 2).unsqueeze(2),
                last_vh.unsqueeze(1).unsqueeze(2),
            ],
            dim=1,
        )  # [B, T+1, 1, NH]
        bTp1_Vl = torch.cat([T_vl.permute(1, 0), last_vl.unsqueeze(1)], dim=1)  # [B, T+1]

        Qh, Ql = compute_dec_ocp_gae(
            Tah_hs=bTah_hs,
            T_l=bT_l,
            Tp1ah_Vh=bTp1ah_Vh,
            Tp1_Vl=bTp1_Vl,
            disc_gamma=discount_factor,
            gae_lambda=gae_lambda,
        )
        adv = compute_cbf_advantages(
            bT_Ql=Ql,
            bT_Vl=T_vl.permute(1, 0),
            bTah_Vh=T_vh.permute(1, 0, 2).unsqueeze(2),
            bTp1ah_Vh=bTp1ah_Vh,
            alpha=alpha,
            cbf_eps=cbf_eps,
            cbf_weight=cbf_weight,
            dt=cbf_dt,
        )

        flat_obs = T_obs.reshape(-1, obs_dim)
        flat_actions = T_actions.reshape(-1, act_dim)
        flat_old_logp = T_logp.reshape(-1)
        flat_adv = adv["bTa_A"].squeeze(-1).reshape(-1)
        flat_Ql = Ql.reshape(-1)
        flat_Qh = Qh.squeeze(2).reshape(-1, n_constraint_heads)

        # Stabilize PPO updates with a global normalization after CBF mixing.
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)

        stats_loss = {
            "total": 0.0,
            "policy": 0.0,
            "value_l": 0.0,
            "value_h": 0.0,
            "entropy": 0.0,
            "clip_frac": 0.0,
            "count": 0,
        }

        total_samples = flat_obs.shape[0]
        for _ in range(learning_epochs):
            perm = torch.randperm(total_samples, device=device)
            for start in range(0, total_samples, mb_size):
                idx = perm[start : start + mb_size]
                mb_obs = flat_obs[idx]
                mb_actions = flat_actions[idx]
                mb_old_logp = flat_old_logp[idx]
                mb_adv = flat_adv[idx].unsqueeze(-1)
                mb_Ql = flat_Ql[idx]
                mb_Qh = flat_Qh[idx]

                dist = policy.dist(mb_obs)
                mb_new_logp = dist.log_prob(mb_actions).sum(dim=-1)
                ratio = torch.exp(mb_new_logp - mb_old_logp)
                ppo_out = compute_policy_surrogate(ratio=ratio.unsqueeze(-1), advantages=mb_adv, clip_eps=ratio_clip)
                loss_policy = ppo_out["loss_policy"]

                pred_vl = value_l(mb_obs).squeeze(-1)
                pred_vh = value_h(mb_obs)
                loss_value_l = nn.functional.mse_loss(pred_vl, mb_Ql)
                loss_value_h = nn.functional.mse_loss(pred_vh, mb_Qh)
                entropy = dist.entropy().sum(dim=-1).mean()

                loss_total = (
                    loss_policy
                    + value_loss_scale * (loss_value_l + value_h_loss_scale * loss_value_h)
                    - entropy_loss_scale * entropy
                )

                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_l.parameters()) + list(value_h.parameters()),
                    grad_norm_clip,
                )
                optimizer.step()

                stats_loss["total"] += float(loss_total.item())
                stats_loss["policy"] += float(loss_policy.item())
                stats_loss["value_l"] += float(loss_value_l.item())
                stats_loss["value_h"] += float(loss_value_h.item())
                stats_loss["entropy"] += float(entropy.item())
                stats_loss["clip_frac"] += float(ppo_out["clip_frac"].item())
                stats_loss["count"] += 1

        denom = max(1, stats_loss["count"])
        stats = DGPPOTrainStats(
            update=update_idx,
            timesteps=timesteps,
            loss_total=stats_loss["total"] / denom,
            loss_policy=stats_loss["policy"] / denom,
            loss_value_l=stats_loss["value_l"] / denom,
            loss_value_h=stats_loss["value_h"] / denom,
            entropy=stats_loss["entropy"] / denom,
            clip_frac=stats_loss["clip_frac"] / denom,
            reward_mean=float(T_rewards.mean().item()),
            cbf_unsafe_frac=float((~adv["bTa_is_safe"]).float().mean().item()),
        )
        print(
            "[DGPPO] update={}/{} timesteps={} reward={:.4f} "
            "loss={:.4f} policy={:.4f} Vl={:.4f} Vh={:.4f} "
            "entropy={:.4f} clip_frac={:.4f} unsafe_frac={:.4f}".format(
                stats.update,
                updates_total,
                stats.timesteps,
                stats.reward_mean,
                stats.loss_total,
                stats.loss_policy,
                stats.loss_value_l,
                stats.loss_value_h,
                stats.entropy,
                stats.clip_frac,
                stats.cbf_unsafe_frac,
            )
        )

        checkpoint_file = checkpoints_dir / f"dgppo_update_{update_idx:06d}.pt"
        torch.save(
            {
                "policy": policy.state_dict(),
                "value_l": value_l.state_dict(),
                "value_h": value_h.state_dict(),
                "optimizer": optimizer.state_dict(),
                "timesteps": timesteps,
                "update": update_idx,
            },
            checkpoint_file,
        )

    final_file = checkpoints_dir / "dgppo_final.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "value_l": value_l.state_dict(),
            "value_h": value_h.state_dict(),
            "optimizer": optimizer.state_dict(),
            "timesteps": timesteps,
            "update": update_idx,
        },
        final_file,
    )
    print(f"[INFO] DGPPO training complete. Final checkpoint: {final_file}")
