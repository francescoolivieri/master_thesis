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
    terminated     (T, B)
    truncated      (T, B)
    rnn_state      one policy carry per rollout step, env, and agent

The final next-observation graph is stored separately so value targets can be
recomputed with current network parameters.
"""

from __future__ import annotations

from skrl.memories.torch import RandomMemory

import torch


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
        use_vl_rnn=False,
        rnn_layers=1,
        rnn_hidden=64,
        rnn_cell="gru",
    ):
        super().__init__(memory_size=rollout_length, num_envs=1, device=device)

        self.n_det_envs = int(num_det_envs)
        self.n_stc_envs = int(num_stc_envs)
        self.rollout_length = int(rollout_length)
        self.use_rnn = bool(use_rnn)
        self.use_vl_rnn = bool(use_vl_rnn)
        self._n_agents = int(n_agents)
        self._n_obs = int(n_obs)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._n_constraints = int(n_constraints)

        # RNN state dimensions
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.rnn_carries = 1 if rnn_cell == "gru" else 2

        num_agents = self._n_agents
        num_obs = self._n_obs
        state_dim = self._state_dim
        action_dim = self._action_dim
        num_constraints = self._n_constraints

        for split, num_envs in (("stc", self.n_stc_envs), ("det", self.n_det_envs)):
            self.create_tensor(
                f"{split}_agent_state",
                size=num_envs * num_agents * state_dim,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            self.create_tensor(
                f"{split}_goal_state",
                size=num_envs * num_agents * state_dim,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            self.create_tensor(
                f"{split}_obs_state",
                size=num_envs * num_obs * state_dim,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            self.create_tensor(
                f"{split}_actions",
                size=num_envs * num_agents * action_dim,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            self.create_tensor(
                f"{split}_log_probs",
                size=num_envs * num_agents,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            self.create_tensor(f"{split}_rewards", size=num_envs, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(
                f"{split}_costs",
                size=num_envs * num_agents * num_constraints,
                dtype=torch.float32,
                keep_dimensions=False,
            )
            # Keep termination and truncation separate for target bootstrap semantics.
            self.create_tensor(f"{split}_terminated", size=num_envs, dtype=torch.bool, keep_dimensions=False)
            self.create_tensor(f"{split}_truncated", size=num_envs, dtype=torch.bool, keep_dimensions=False)

            if self.use_rnn:
                rnn_size = num_envs * num_agents * self.rnn_layers * self.rnn_carries * self.rnn_hidden
                self.create_tensor(f"{split}_rnn_states", size=rnn_size, dtype=torch.float32, keep_dimensions=False)
        self._initial_vl_rnn_states = None
        if self.use_vl_rnn:
            self._initial_vl_rnn_states = {
                "stc": torch.zeros(
                    self.n_stc_envs,
                    self.rnn_layers,
                    self.rnn_carries,
                    self.rnn_hidden,
                    dtype=torch.float32,
                    device=device,
                ),
                "det": torch.zeros(
                    self.n_det_envs,
                    self.rnn_layers,
                    self.rnn_carries,
                    self.rnn_hidden,
                    dtype=torch.float32,
                    device=device,
                ),
            }
        self._final_agent_state = {
            "stc": torch.zeros(self.n_stc_envs, num_agents, state_dim, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, num_agents, state_dim, dtype=torch.float32, device=device),
        }
        self._final_goal_state = {
            "stc": torch.zeros(self.n_stc_envs, num_agents, state_dim, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, num_agents, state_dim, dtype=torch.float32, device=device),
        }
        self._final_obs_state = {
            "stc": torch.zeros(self.n_stc_envs, num_obs, state_dim, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, num_obs, state_dim, dtype=torch.float32, device=device),
        }
        self._cursor = 0

    def _tensor(self, name: str) -> torch.Tensor:
        """Return the flat per-step storage tensor created by skrl memory."""
        return self.tensors[name].squeeze(1)

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
        det_agent_state: torch.Tensor,
        det_goal_state: torch.Tensor,
        det_obs_state: torch.Tensor,
        det_action: torch.Tensor,
        det_log_prob: torch.Tensor,
        det_reward: torch.Tensor,
        det_cost: torch.Tensor,
        stc_terminated: torch.Tensor | None = None,
        stc_truncated: torch.Tensor | None = None,
        det_terminated: torch.Tensor | None = None,
        det_truncated: torch.Tensor | None = None,
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
        self._tensor("stc_costs")[t] = stc_cost.reshape(-1)
        self._tensor("stc_terminated")[t] = self._canonical_done_mask(stc_terminated, self.n_stc_envs)
        self._tensor("stc_truncated")[t] = self._canonical_done_mask(stc_truncated, self.n_stc_envs)

        self._tensor("det_agent_state")[t] = det_agent_state.reshape(-1)
        self._tensor("det_goal_state")[t] = det_goal_state.reshape(-1)
        self._tensor("det_obs_state")[t] = det_obs_state.reshape(-1)
        self._tensor("det_actions")[t] = det_action.reshape(-1)
        self._tensor("det_log_probs")[t] = det_log_prob.reshape(-1)
        self._tensor("det_rewards")[t] = det_reward.reshape(-1)
        self._tensor("det_costs")[t] = det_cost.reshape(-1)
        self._tensor("det_terminated")[t] = self._canonical_done_mask(det_terminated, self.n_det_envs)
        self._tensor("det_truncated")[t] = self._canonical_done_mask(det_truncated, self.n_det_envs)

        if self.use_rnn:
            if stc_rnn_state is not None:
                self._tensor("stc_rnn_states")[t] = self._canonical_rnn_state(
                    stc_rnn_state, self.n_stc_envs
                ).reshape(-1)
            if det_rnn_state is not None:
                self._tensor("det_rnn_states")[t] = self._canonical_rnn_state(
                    det_rnn_state, self.n_det_envs
                ).reshape(-1)
        self._cursor += 1

    def _canonical_rnn_state(self, rnn_state: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Return policy carry with env and agent as the first two dimensions."""
        num_agents = self._n_agents
        flat_shape = (self.rnn_layers, num_envs * num_agents, self.rnn_carries, self.rnn_hidden)
        stored_shape = (num_envs, num_agents, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
        if rnn_state.shape == flat_shape:
            return rnn_state.reshape(
                self.rnn_layers,
                num_envs,
                num_agents,
                self.rnn_carries,
                self.rnn_hidden,
            ).permute(1, 2, 0, 3, 4)
        if rnn_state.shape == stored_shape:
            return rnn_state
        raise ValueError(
            "Unexpected RNN state shape "
            f"{tuple(rnn_state.shape)}; expected "
            f"{flat_shape} or {stored_shape}"
        )

    def _canonical_vl_rnn_state(self, rnn_state: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Return centralized Vl carry with env as the first dimension."""
        expected_flat = (self.rnn_layers, num_envs, self.rnn_carries, self.rnn_hidden)
        expected_stored = (num_envs, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
        if rnn_state.shape == expected_flat:
            return rnn_state.permute(1, 0, 2, 3)
        if rnn_state.shape == expected_stored:
            return rnn_state
        raise ValueError(
            "Unexpected Vl RNN state shape "
            f"{tuple(rnn_state.shape)}; expected {expected_flat} or {expected_stored}"
        )

    def _canonical_done_mask(self, mask: torch.Tensor | None, num_envs: int) -> torch.Tensor:
        """Return a flat boolean mask of length ``num_envs`` for one rollout split."""
        if mask is None:
            return torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        return torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(num_envs)

    def _num_envs_for_split(self, split: str) -> int:
        if split == "stc":
            return self.n_stc_envs
        if split == "det":
            return self.n_det_envs
        raise ValueError(f"Unknown split '{split}'")

    def set_final_state(
        self,
        split: str,
        *,
        agent_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs_state: torch.Tensor,
    ) -> None:
        """Store the final next-observation graph."""
        num_envs = self._num_envs_for_split(split)
        self._final_agent_state[split] = agent_state.reshape(
            num_envs, self._n_agents, self._state_dim
        ).to(self.device)
        self._final_goal_state[split] = goal_state.reshape(
            num_envs, self._n_agents, self._state_dim
        ).to(self.device)
        self._final_obs_state[split] = obs_state.reshape(num_envs, self._n_obs, self._state_dim).to(self.device)

    def set_initial_vl_state(self, split: str, rnn_state: torch.Tensor | None) -> None:
        """Store the centralized critic carry at the first rollout step."""
        if rnn_state is None or not self.use_vl_rnn:
            return
        num_envs = self._num_envs_for_split(split)
        assert self._initial_vl_rnn_states is not None
        self._initial_vl_rnn_states[split] = self._canonical_vl_rnn_state(rnn_state, num_envs).to(self.device)

    # Read-side helpers used by the update
    # ------------------------------------------------------------------

    def as_update_view(self, split: str) -> dict[str, torch.Tensor]:
        """Return a rollout split in the ``[B, T, ...]`` layout used by the update."""
        num_envs = self._num_envs_for_split(split)
        num_agents = self._n_agents
        num_obs = self._n_obs
        rollout_length = self.rollout_length

        agent_state = self._tensor(f"{split}_agent_state").reshape(
            rollout_length, num_envs, num_agents, self._state_dim
        )
        goal_state = self._tensor(f"{split}_goal_state").reshape(
            rollout_length, num_envs, num_agents, self._state_dim
        )
        obs_state = self._tensor(f"{split}_obs_state").reshape(rollout_length, num_envs, num_obs, self._state_dim)
        actions = self._tensor(f"{split}_actions").reshape(rollout_length, num_envs, num_agents, self._action_dim)
        log_probs = self._tensor(f"{split}_log_probs").reshape(rollout_length, num_envs, num_agents)
        rewards = self._tensor(f"{split}_rewards").reshape(rollout_length, num_envs)
        costs = self._tensor(f"{split}_costs").reshape(
            rollout_length, num_envs, num_agents, self._n_constraints
        )
        terminated = self._tensor(f"{split}_terminated").reshape(rollout_length, num_envs)
        truncated = self._tensor(f"{split}_truncated").reshape(rollout_length, num_envs)

        data = {
            "bT_l": -rewards.transpose(0, 1),
            "bTah_hs": costs.transpose(0, 1),
            "bT_terminated": terminated.transpose(0, 1),
            "bT_truncated": truncated.transpose(0, 1),
            "bT_done": (terminated | truncated).transpose(0, 1),
            "bTa_logp": log_probs.transpose(0, 1),
            "bTa_actions": actions.transpose(0, 1),
            "bTa_agent_state": agent_state.transpose(0, 1),
            "bTa_goal_state": goal_state.transpose(0, 1),
            "bTo_obs_state": obs_state.transpose(0, 1),
            "b_final_agent_state": self._final_agent_state[split],
            "b_final_goal_state": self._final_goal_state[split],
            "b_final_obs_state": self._final_obs_state[split],
        }

        if self.use_rnn:
            rnn_states = self._tensor(f"{split}_rnn_states").reshape(
                rollout_length,
                num_envs,
                num_agents,
                self.rnn_layers,
                self.rnn_carries,
                self.rnn_hidden,
            )
            data["bTa_rnn_states"] = rnn_states.permute(1, 0, 3, 2, 4, 5)
        if self.use_vl_rnn:
            assert self._initial_vl_rnn_states is not None
            data["b_initial_vl_rnn_state"] = self._initial_vl_rnn_states[split]

        return data

    def as_bTah_view(self, split: str) -> dict[str, torch.Tensor]:
        """Compatibility alias for older tests and scripts."""
        return self.as_update_view(split)

    def sample_minibatches(self, num_mini_batches: int) -> list[torch.Tensor]:
        """Return randomized chunks of env indices over the stochastic split."""
        if num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be > 0, got {num_mini_batches}")
        num_envs = self.n_stc_envs
        perm = torch.randperm(num_envs, device=self.device)
        chunks = torch.tensor_split(perm, num_mini_batches)
        return [env_ids for env_ids in chunks if env_ids.numel() > 0]

    def minibatch_iter(self, num_mini_batches: int):
        """Yield randomized chunks of env indices over the stochastic split."""
        for env_ids in self.sample_minibatches(num_mini_batches):
            yield env_ids
