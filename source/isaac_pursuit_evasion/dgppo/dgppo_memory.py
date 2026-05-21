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

from typing import Optional
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

        S, A, O, NH, Da = state_dim, n_agents, n_obs, n_constraints, action_dim
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

        for prefix, B in (("stc", self.n_stc_envs), ("det", self.n_det_envs)):
            self.create_tensor(f"{prefix}_agent_state", size=B * A * S, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_goal_state", size=B * A * S, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_obs_state", size=B * O * S, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_actions", size=B * A * Da, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_log_probs", size=B * A, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_rewards", size=B, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_costs", size=B * A * NH, dtype=torch.float32, keep_dimensions=False)
            # Keep termination and truncation separate for target bootstrap semantics.
            self.create_tensor(f"{prefix}_terminated", size=B, dtype=torch.bool, keep_dimensions=False)
            self.create_tensor(f"{prefix}_truncated", size=B, dtype=torch.bool, keep_dimensions=False)

            if self.use_rnn:
                rnn_size = B * A * self.rnn_layers * self.rnn_carries * self.rnn_hidden
                self.create_tensor(f"{prefix}_rnn_states", size=rnn_size, dtype=torch.float32, keep_dimensions=False)
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
            "stc": torch.zeros(self.n_stc_envs, A, S, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, A, S, dtype=torch.float32, device=device),
        }
        self._final_goal_state = {
            "stc": torch.zeros(self.n_stc_envs, A, S, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, A, S, dtype=torch.float32, device=device),
        }
        self._final_obs_state = {
            "stc": torch.zeros(self.n_stc_envs, O, S, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, O, S, dtype=torch.float32, device=device),
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
        stc_terminated: Optional[torch.Tensor] = None,
        stc_truncated: Optional[torch.Tensor] = None,
        det_terminated: Optional[torch.Tensor] = None,
        det_truncated: Optional[torch.Tensor] = None,
        stc_rnn_state: Optional[torch.Tensor] = None,
        det_rnn_state: Optional[torch.Tensor] = None,
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

    def _canonical_rnn_state(self, rnn_state: torch.Tensor, n_envs: int) -> torch.Tensor:
        """Return policy carry with env and agent as the first two dimensions."""
        A = self._n_agents
        flat_shape = (self.rnn_layers, n_envs * A, self.rnn_carries, self.rnn_hidden)
        stored_shape = (n_envs, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
        if rnn_state.shape == flat_shape:
            return rnn_state.reshape(
                self.rnn_layers,
                n_envs,
                A,
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

    def _canonical_vl_rnn_state(self, rnn_state: torch.Tensor, n_envs: int) -> torch.Tensor:
        """Return centralized Vl carry with env as the first dimension."""
        expected_flat = (self.rnn_layers, n_envs, self.rnn_carries, self.rnn_hidden)
        expected_stored = (n_envs, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
        if rnn_state.shape == expected_flat:
            return rnn_state.permute(1, 0, 2, 3)
        if rnn_state.shape == expected_stored:
            return rnn_state
        raise ValueError(
            "Unexpected Vl RNN state shape "
            f"{tuple(rnn_state.shape)}; expected {expected_flat} or {expected_stored}"
        )

    def _canonical_done_mask(self, mask: Optional[torch.Tensor], B: int) -> torch.Tensor:
        """Return a flat boolean mask of length ``B`` for one rollout split."""
        if mask is None:
            return torch.zeros(B, dtype=torch.bool, device=self.device)
        return torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(B)

    def set_final_state(
        self,
        split: str,
        *,
        agent_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs_state: torch.Tensor,
    ) -> None:
        """Store the final next-observation graph."""
        if split not in ("stc", "det"):
            raise ValueError(f"Unknown split '{split}'")
        B = self.n_stc_envs if split == "stc" else self.n_det_envs
        self._final_agent_state[split] = agent_state.reshape(B, self._n_agents, self._state_dim).to(self.device)
        self._final_goal_state[split] = goal_state.reshape(B, self._n_agents, self._state_dim).to(self.device)
        self._final_obs_state[split] = obs_state.reshape(B, self._n_obs, self._state_dim).to(self.device)

    def set_initial_vl_state(self, split: str, rnn_state: Optional[torch.Tensor]) -> None:
        """Store the centralized critic carry at the first rollout step."""
        if rnn_state is None or not self.use_vl_rnn:
            return
        if split not in ("stc", "det"):
            raise ValueError(f"Unknown split '{split}'")
        B = self.n_stc_envs if split == "stc" else self.n_det_envs
        assert self._initial_vl_rnn_states is not None
        self._initial_vl_rnn_states[split] = self._canonical_vl_rnn_state(rnn_state, B).to(self.device)

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
        O = self._n_obs
        T = self.rollout_length

        agent_state = self._tensor(f"{split}_agent_state").reshape(T, B, A, self._state_dim)
        goal_state = self._tensor(f"{split}_goal_state").reshape(T, B, A, self._state_dim)
        obs_state = self._tensor(f"{split}_obs_state").reshape(T, B, O, self._state_dim)
        actions = self._tensor(f"{split}_actions").reshape(T, B, A, self._action_dim)
        log_probs = self._tensor(f"{split}_log_probs").reshape(T, B, A)
        rewards = self._tensor(f"{split}_rewards").reshape(T, B)
        costs = self._tensor(f"{split}_costs").reshape(T, B, A, self._n_constraints)
        terminated = self._tensor(f"{split}_terminated").reshape(T, B)
        truncated = self._tensor(f"{split}_truncated").reshape(T, B)

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
                T, B, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden
            )
            data["bTa_rnn_states"] = rnn_states.permute(1, 0, 3, 2, 4, 5)
        if self.use_vl_rnn:
            assert self._initial_vl_rnn_states is not None
            data["b_initial_vl_rnn_state"] = self._initial_vl_rnn_states[split]

        return data

    def sample_minibatches(self, num_mini_batches: int) -> list[torch.Tensor]:
        """Return randomized chunks of env indices over the stochastic split.
        """
        if num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be > 0, got {num_mini_batches}")
        B = self.n_stc_envs
        perm = torch.randperm(B, device=self.device)
        chunks = torch.tensor_split(perm, num_mini_batches)
        return [idx for idx in chunks if idx.numel() > 0]

    def minibatch_iter(self, num_mini_batches: int):
        """Yield randomized chunks of env indices over the stochastic split."""
        for idx in self.sample_minibatches(num_mini_batches):
            yield idx
