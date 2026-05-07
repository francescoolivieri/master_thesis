"""
Rollout memory used by DGPPOAgent.


    T: rollout length
    B: num parallel environments
    A: num agents
    NH: num constraint heads
    D_a: action dim
    D_s: state dim per node

    agent_state    (T, B, A, D_s)
    goal_state     (T, B, A, D_s)   -- one goal per agent
    obs_state      (T, B, O, D_s)
    actions        (T, B, A, D_a)
    log_prob       (T, B, A)
    rewards        (T, B)
    costs          (T, B, A, NH)
    values_l       (T+1, B)
    values_h       (T+1, B, A, NH)
    rnn_state      (T, B, A, L, C, H)  -- only kept if the policy has an RNN

``values_l`` / ``values_h`` have one extra step for the terminal bootstrap.
"""

from __future__ import annotations

import torch
from skrl.memories.torch import RandomMemory


class DGPPORolloutMemory(RandomMemory):

    def __init__(
        self,
        rollout_length,
        num_det_envs,
        num_stc_envs,
        n_agents,
        n_obs,
        state_dim,
        action_dim,
        n_constraints,
        device,
        use_rnn=False,
        rnn_layers=1,
        rnn_hidden=64,
        rnn_cell="gru",
    ):
        super().__init__(memory_size=rollout_length, num_envs=1, device=device)

        S, A, n_obs_local, NH, Da = state_dim, n_agents, n_obs, n_constraints, action_dim
        self.n_det_envs = int(num_det_envs)
        self.n_stc_envs = int(num_stc_envs)
        self.rollout_length = int(rollout_length)
        self.use_rnn = bool(use_rnn)
        self._n_agents = int(n_agents)
        self._n_obs = int(n_obs)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._n_constraints = int(n_constraints)

        # RNN state dimensions
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.rnn_carries = 1 if rnn_cell == "gru" else 2

        for prefix, B in (("stc", self.n_stc_envs), ("det", self.n_det_envs)):
            self.create_tensor(f"{prefix}_agent_state", size=B * A * S, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_goal_state", size=B * A * S, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(
                f"{prefix}_obs_state", size=B * n_obs_local * S, dtype=torch.float32, keep_dimensions=False
            )
            self.create_tensor(f"{prefix}_actions", size=B * A * Da, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_log_probs", size=B * A, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_rewards", size=B, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_dones", size=B, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_costs", size=B * A * NH, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_values_l", size=B, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_values_h", size=B * A * NH, dtype=torch.float32, keep_dimensions=False)

            if self.use_rnn:
                # [B * A, L, C, H] flattened for RandomMemory which expects a flat size per step
                rnn_size = B * A * self.rnn_layers * self.rnn_carries * self.rnn_hidden
                self.create_tensor(f"{prefix}_rnn_states", size=rnn_size, dtype=torch.float32, keep_dimensions=False)

        self._final_values_l = {
            "stc": torch.zeros(self.n_stc_envs, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, dtype=torch.float32, device=device),
        }
        self._final_values_h = {
            "stc": torch.zeros(self.n_stc_envs, A, NH, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, A, NH, dtype=torch.float32, device=device),
        }
        self._cursor = 0

    def _tensor(self, name: str) -> torch.Tensor:
        """Return the flat per-step storage tensor created by skrl memory."""
        return self.tensors[name].squeeze(1)

    # ------------------------------------------------------------------
    # Check from here
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._cursor = 0

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.rollout_length

    @property
    def cursor(self) -> int:
        return self._cursor

    def add(
        self,
        *,
        stc_agent_state: torch.Tensor,
        stc_goal_state: torch.Tensor,
        stc_obs_state: torch.Tensor,
        stc_action: torch.Tensor,
        stc_log_prob: torch.Tensor,
        stc_reward: torch.Tensor,
        stc_cost: torch.Tensor,
        stc_value_l: torch.Tensor,
        stc_value_h: torch.Tensor,
        det_agent_state: torch.Tensor,
        det_goal_state: torch.Tensor,
        det_obs_state: torch.Tensor,
        det_action: torch.Tensor,
        det_log_prob: torch.Tensor,
        det_reward: torch.Tensor,
        det_cost: torch.Tensor,
        det_value_l: torch.Tensor,
        det_value_h: torch.Tensor,
        stc_done: torch.Tensor | None = None,
        det_done: torch.Tensor | None = None,
        stc_rnn_state: torch.Tensor | None = None,
        det_rnn_state: torch.Tensor | None = None,
    ) -> None:
        """Append one rollout step for both stochastic and deterministic splits."""
        t = self._cursor
        if t >= self.rollout_length:
            raise RuntimeError("DGPPORolloutMemory.add called past rollout_length")

        self._tensor("stc_agent_state")[t] = stc_agent_state.reshape(-1)
        self._tensor("stc_goal_state")[t] = stc_goal_state.reshape(-1)
        self._tensor("stc_obs_state")[t] = stc_obs_state.reshape(-1)
        self._tensor("stc_actions")[t] = stc_action.reshape(-1)
        self._tensor("stc_log_probs")[t] = stc_log_prob.reshape(-1)
        self._tensor("stc_rewards")[t] = stc_reward.reshape(-1)
        if stc_done is None:
            self._tensor("stc_dones")[t].zero_()
        else:
            self._tensor("stc_dones")[t] = stc_done.to(dtype=torch.float32).reshape(-1)
        self._tensor("stc_costs")[t] = stc_cost.reshape(-1)
        self._tensor("stc_values_l")[t] = stc_value_l.reshape(-1)
        self._tensor("stc_values_h")[t] = stc_value_h.reshape(-1)

        self._tensor("det_agent_state")[t] = det_agent_state.reshape(-1)
        self._tensor("det_goal_state")[t] = det_goal_state.reshape(-1)
        self._tensor("det_obs_state")[t] = det_obs_state.reshape(-1)
        self._tensor("det_actions")[t] = det_action.reshape(-1)
        self._tensor("det_log_probs")[t] = det_log_prob.reshape(-1)
        self._tensor("det_rewards")[t] = det_reward.reshape(-1)
        if det_done is None:
            self._tensor("det_dones")[t].zero_()
        else:
            self._tensor("det_dones")[t] = det_done.to(dtype=torch.float32).reshape(-1)
        self._tensor("det_costs")[t] = det_cost.reshape(-1)
        self._tensor("det_values_l")[t] = det_value_l.reshape(-1)
        self._tensor("det_values_h")[t] = det_value_h.reshape(-1)

        if self.use_rnn:
            if stc_rnn_state is not None:
                self._tensor("stc_rnn_states")[t] = self._canonical_rnn_state(stc_rnn_state, self.n_stc_envs).reshape(
                    -1
                )
            if det_rnn_state is not None:
                self._tensor("det_rnn_states")[t] = self._canonical_rnn_state(det_rnn_state, self.n_det_envs).reshape(
                    -1
                )

        self._cursor += 1

    def _canonical_rnn_state(self, rnn_state: torch.Tensor, B: int) -> torch.Tensor:
        """Return policy RNN state as ``[B, A, L, C, H]`` for storage."""
        A = self._n_agents
        if rnn_state.shape == (self.rnn_layers, B * A, self.rnn_carries, self.rnn_hidden):
            return rnn_state.reshape(self.rnn_layers, B, A, self.rnn_carries, self.rnn_hidden).permute(1, 2, 0, 3, 4)
        if rnn_state.shape == (B, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden):
            return rnn_state
        raise ValueError(
            "Unexpected RNN state shape "
            f"{tuple(rnn_state.shape)}; expected "
            f"{(self.rnn_layers, B * A, self.rnn_carries, self.rnn_hidden)} or "
            f"{(B, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden)}"
        )

    def set_final_values(self, split: str, value_l: torch.Tensor, value_h: torch.Tensor) -> None:
        if split not in ("stc", "det"):
            raise ValueError(f"Unknown split '{split}'")
        self._final_values_l[split] = value_l.reshape(-1).to(self.device)
        self._final_values_h[split] = value_h.to(self.device)

    # Read-side helpers used by the update
    # ------------------------------------------------------------------

    def as_bTah_view(self, split: str) -> dict[str, torch.Tensor]:
        """Return tensors transposed to the ``(B, T, ...)`` layout expected
        by :func:`train_dgppo.compute_dec_ocp_gae` /
        :func:`compute_cbf_advantages`.
        """
        if split not in ("stc", "det"):
            raise ValueError(f"Unknown split '{split}'")
        B = self.n_stc_envs if split == "stc" else self.n_det_envs
        A = self._n_agents
        n_obs = self._n_obs
        T = self.rollout_length

        agent_state = self._tensor(f"{split}_agent_state").reshape(T, B, A, self._state_dim)
        goal_state = self._tensor(f"{split}_goal_state").reshape(T, B, A, self._state_dim)
        obs_state = self._tensor(f"{split}_obs_state").reshape(T, B, n_obs, self._state_dim)
        actions = self._tensor(f"{split}_actions").reshape(T, B, A, self._action_dim)
        log_probs = self._tensor(f"{split}_log_probs").reshape(T, B, A)
        rewards = self._tensor(f"{split}_rewards").reshape(T, B)
        dones = self._tensor(f"{split}_dones").reshape(T, B)
        costs = self._tensor(f"{split}_costs").reshape(T, B, A, self._n_constraints)
        values_l = self._tensor(f"{split}_values_l").reshape(T, B)
        values_h = self._tensor(f"{split}_values_h").reshape(T, B, A, self._n_constraints)
        v_l_tp1 = torch.cat([values_l, self._final_values_l[split].unsqueeze(0)], dim=0)
        v_h_tp1 = torch.cat([values_h, self._final_values_h[split].unsqueeze(0)], dim=0)

        data = {
            "bT_l": -rewards.transpose(0, 1),
            "bT_dones": dones.transpose(0, 1).to(dtype=torch.bool),
            "bTah_hs": costs.transpose(0, 1),
            "bTp1_Vl": v_l_tp1.transpose(0, 1),
            "bTp1ah_Vh": v_h_tp1.transpose(0, 1),
            "bTah_Vh": values_h.transpose(0, 1),
            "bT_Vl": values_l.transpose(0, 1),
            "bTa_logp": log_probs.transpose(0, 1),
            "bTa_actions": actions.transpose(0, 1),
            "bTa_agent_state": agent_state.transpose(0, 1),
            "bTa_goal_state": goal_state.transpose(0, 1),
            "bTo_obs_state": obs_state.transpose(0, 1),
        }

        if self.use_rnn:
            # RNN state shape in memory: [T, B * A * L * C * H]
            # Transpose to [B, T, L, A, C, H], matching the RNN carry layout per rollout step.
            # Wait, N = B * A. The memory stores it as [T, N * L * C * H]
            rnn_states = self._tensor(f"{split}_rnn_states").reshape(
                T, B, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden
            )
            data["bTa_rnn_states"] = rnn_states.permute(1, 0, 3, 2, 4, 5)  # [B, T, L, A, C, H]

        return data

    def sample_minibatches(self, num_mini_batches: int) -> list[torch.Tensor]:
        """Return randomized chunks of env indices over the stochastic split.

        DGPPO's PPO loop shuffles env indices and passes ``B/num_minibatches``
        trajectories per step, matching the JAX reference
        (``update_inner`` in ``dgppo/algo/dgppo.py``).
        """
        if num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be > 0, got {num_mini_batches}")
        B = self.n_stc_envs
        perm = torch.randperm(B, device=self.device)
        chunks = torch.tensor_split(perm, num_mini_batches)
        return [idx for idx in chunks if idx.numel() > 0]

    def minibatch_iter(self, num_mini_batches: int):
        """Yield randomized chunks of env indices over the stochastic split."""
        yield from self.sample_minibatches(num_mini_batches)
