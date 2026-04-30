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
    rnn_state      (T, B, A, L, C, H) -- policy/Vh carry, only kept with RNNs
    vl_rnn_state   (T, B, L, C, H)    -- centralized Vl carry, only kept with RNNs

``values_l`` / ``values_h`` have one extra step for the terminal bootstrap.
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
        self._n_agents = int(n_agents)
        self._n_obs = int(n_obs)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._n_constraints = int(n_constraints)
        self._cursor = 0
        
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
            self.create_tensor(f"{prefix}_values_l", size=B, dtype=torch.float32, keep_dimensions=False)
            self.create_tensor(f"{prefix}_values_h", size=B * A * NH, dtype=torch.float32, keep_dimensions=False)
            
            if self.use_rnn:
                # [B * A, L, C, H] flattened for RandomMemory which expects a flat size per step
                rnn_size = B * A * self.rnn_layers * self.rnn_carries * self.rnn_hidden
                self.create_tensor(f"{prefix}_rnn_states", size=rnn_size, dtype=torch.float32, keep_dimensions=False)
                vl_rnn_size = B * self.rnn_layers * self.rnn_carries * self.rnn_hidden
                self.create_tensor(f"{prefix}_vl_rnn_states", size=vl_rnn_size, dtype=torch.float32, keep_dimensions=False)

        self._final_values_l = {
            "stc": torch.zeros(self.n_stc_envs, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, dtype=torch.float32, device=device),
        }
        self._final_values_h = {
            "stc": torch.zeros(self.n_stc_envs, A, NH, dtype=torch.float32, device=device),
            "det": torch.zeros(self.n_det_envs, A, NH, dtype=torch.float32, device=device),
        }

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
        stc_rnn_state: Optional[torch.Tensor] = None,
        det_rnn_state: Optional[torch.Tensor] = None,
        stc_vl_rnn_state: Optional[torch.Tensor] = None,
        det_vl_rnn_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one rollout step for both stochastic and deterministic splits."""
        t = self._cursor
        if t >= self.rollout_length:
            raise RuntimeError("DGPPORolloutMemory.add called past rollout_length")

        self.tensors["stc_agent_state"][t] = stc_agent_state.reshape(-1)
        self.tensors["stc_goal_state"][t] = stc_goal_state.reshape(-1)
        self.tensors["stc_obs_state"][t] = stc_obs_state.reshape(-1)
        self.tensors["stc_actions"][t] = stc_action.reshape(-1)
        self.tensors["stc_log_probs"][t] = stc_log_prob.reshape(-1)
        self.tensors["stc_rewards"][t] = stc_reward.reshape(-1)
        self.tensors["stc_costs"][t] = stc_cost.reshape(-1)
        self.tensors["stc_values_l"][t] = stc_value_l.reshape(-1)
        self.tensors["stc_values_h"][t] = stc_value_h.reshape(-1)

        self.tensors["det_agent_state"][t] = det_agent_state.reshape(-1)
        self.tensors["det_goal_state"][t] = det_goal_state.reshape(-1)
        self.tensors["det_obs_state"][t] = det_obs_state.reshape(-1)
        self.tensors["det_actions"][t] = det_action.reshape(-1)
        self.tensors["det_log_probs"][t] = det_log_prob.reshape(-1)
        self.tensors["det_rewards"][t] = det_reward.reshape(-1)
        self.tensors["det_costs"][t] = det_cost.reshape(-1)
        self.tensors["det_values_l"][t] = det_value_l.reshape(-1)
        self.tensors["det_values_h"][t] = det_value_h.reshape(-1)
        
        if self.use_rnn:
            if stc_rnn_state is not None:
                self.tensors["stc_rnn_states"][t] = stc_rnn_state.reshape(-1)
            if det_rnn_state is not None:
                self.tensors["det_rnn_states"][t] = det_rnn_state.reshape(-1)
            if stc_vl_rnn_state is not None:
                self.tensors["stc_vl_rnn_states"][t] = stc_vl_rnn_state.reshape(-1)
            if det_vl_rnn_state is not None:
                self.tensors["det_vl_rnn_states"][t] = det_vl_rnn_state.reshape(-1)

        self._cursor += 1

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
        O = self._n_obs
        T = self.rollout_length

        agent_state = self.tensors[f"{split}_agent_state"].reshape(T, B, A, self._state_dim)
        goal_state = self.tensors[f"{split}_goal_state"].reshape(T, B, A, self._state_dim)
        obs_state = self.tensors[f"{split}_obs_state"].reshape(T, B, O, self._state_dim)
        actions = self.tensors[f"{split}_actions"].reshape(T, B, A, self._action_dim)
        log_probs = self.tensors[f"{split}_log_probs"].reshape(T, B, A)
        rewards = self.tensors[f"{split}_rewards"].reshape(T, B)
        costs = self.tensors[f"{split}_costs"].reshape(T, B, A, self._n_constraints)
        values_l = self.tensors[f"{split}_values_l"].reshape(T, B)
        values_h = self.tensors[f"{split}_values_h"].reshape(T, B, A, self._n_constraints)
        v_l_tp1 = torch.cat([values_l, self._final_values_l[split].unsqueeze(0)], dim=0)
        v_h_tp1 = torch.cat([values_h, self._final_values_h[split].unsqueeze(0)], dim=0)

        data = {
            "bT_l": -rewards.transpose(0, 1),
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
            rnn_states = self.tensors[f"{split}_rnn_states"].reshape(T, B, A, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
            data["bTa_rnn_states"] = rnn_states.transpose(0, 1) # [B, T, A, L, C, H]
            vl_rnn_states = self.tensors[f"{split}_vl_rnn_states"].reshape(T, B, self.rnn_layers, self.rnn_carries, self.rnn_hidden)
            data["bT_vl_rnn_states"] = vl_rnn_states.transpose(0, 1) # [B, T, L, C, H]
            
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
        for idx in self.sample_minibatches(num_mini_batches):
            yield idx
