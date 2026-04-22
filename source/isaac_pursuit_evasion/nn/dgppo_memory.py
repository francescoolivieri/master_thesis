"""
Rollout memory used by :class:`DGPPOAgent`.

Rather than keeping a list of :class:`GraphData` objects for every step of
every environment (which blows up memory for large ``num_envs`` and ``T``),
we store only the *raw* per-env agent / goal / obstacle state tensors and
rebuild the graph on-demand during the update phase via
:func:`graph_builder.build_graph_data`. The CPU cost of rebuilding is small
(a handful of ``cdist`` + ``torch.cat`` ops) and easily offset by the memory
savings at ``num_envs=4096, T=32``.

Tensor layout follows the JAX reference convention (see
``dgppo/trainer/data.py::Rollout``):

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
    rnn_state      (T, L, B*A, C, H)  -- only kept if the policy has an RNN

``values_l`` / ``values_h`` have one extra step for the terminal bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DGPPORolloutMemory:
    """Fixed-capacity rollout buffer for DGPPO.

    Created once by the agent with the expected shapes; callers use
    :meth:`add` each env step and :meth:`set_final_values` before update.
    Everything lives on ``device`` (typically CUDA) to avoid H2D copies
    during PPO epochs.
    """

    rollout_length: int
    num_envs: int
    n_agents: int
    n_obs: int
    state_dim: int
    action_dim: int
    n_constraints: int
    device: torch.device
    dtype: torch.dtype = torch.float32
    use_rnn: bool = False
    rnn_layers: int = 1
    rnn_carries: int = 1
    rnn_hidden: int = 64

    # Filled by :meth:`allocate`.
    agent_state: torch.Tensor = field(init=False)
    goal_state: torch.Tensor = field(init=False)
    obs_state: torch.Tensor = field(init=False)
    actions: torch.Tensor = field(init=False)
    log_prob: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    costs: torch.Tensor = field(init=False)
    values_l: torch.Tensor = field(init=False)
    values_h: torch.Tensor = field(init=False)
    rnn_state: Optional[torch.Tensor] = field(init=False, default=None)
    _cursor: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        T, B, A, O = self.rollout_length, self.num_envs, self.n_agents, self.n_obs
        S, Da, NH = self.state_dim, self.action_dim, self.n_constraints
        dev, dt = self.device, self.dtype

        self.agent_state = torch.zeros((T, B, A, S), device=dev, dtype=dt)
        self.goal_state = torch.zeros((T, B, A, S), device=dev, dtype=dt)
        self.obs_state = torch.zeros((T, B, O, S), device=dev, dtype=dt)
        self.actions = torch.zeros((T, B, A, Da), device=dev, dtype=dt)
        self.log_prob = torch.zeros((T, B, A), device=dev, dtype=dt)
        self.rewards = torch.zeros((T, B), device=dev, dtype=dt)
        self.costs = torch.zeros((T, B, A, NH), device=dev, dtype=dt)
        self.values_l = torch.zeros((T + 1, B), device=dev, dtype=dt)
        self.values_h = torch.zeros((T + 1, B, A, NH), device=dev, dtype=dt)
        if self.use_rnn:
            self.rnn_state = torch.zeros(
                (T, self.rnn_layers, B * A, self.rnn_carries, self.rnn_hidden),
                device=dev,
                dtype=dt,
            )

    # ------------------------------------------------------------------
    # Write-side API
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
        agent_state: torch.Tensor,   # (B, A, S)
        goal_state: torch.Tensor,    # (B, A, S)
        obs_state: torch.Tensor,     # (B, O, S)
        action: torch.Tensor,        # (B, A, Da) OR (B*A, Da)
        log_prob: torch.Tensor,      # (B, A)     OR (B*A,)
        reward: torch.Tensor,        # (B,)
        cost: torch.Tensor,          # (B, A, NH)
        value_l: torch.Tensor,       # (B,)       OR (B, 1)
        value_h: torch.Tensor,       # (B, A, NH)
        rnn_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Append a single transition row (all env-parallel)."""
        t = self._cursor
        if t >= self.rollout_length:
            raise RuntimeError("DGPPORolloutMemory.add called past rollout_length")

        B, A = self.num_envs, self.n_agents
        self.agent_state[t] = agent_state
        self.goal_state[t] = goal_state
        self.obs_state[t] = obs_state
        self.actions[t] = action.reshape(B, A, -1)
        self.log_prob[t] = log_prob.reshape(B, A)
        self.rewards[t] = reward.reshape(B)
        self.costs[t] = cost
        self.values_l[t] = value_l.reshape(B)
        self.values_h[t] = value_h
        if self.use_rnn and rnn_state is not None and self.rnn_state is not None:
            self.rnn_state[t] = rnn_state
        self._cursor += 1

    def set_final_values(self, value_l: torch.Tensor, value_h: torch.Tensor) -> None:
        """Store the bootstrap values at index ``T`` (one past the last step)."""
        self.values_l[self.rollout_length] = value_l.reshape(self.num_envs)
        self.values_h[self.rollout_length] = value_h

    # ------------------------------------------------------------------
    # Read-side helpers used by the update
    # ------------------------------------------------------------------

    def as_bTah_view(self) -> dict[str, torch.Tensor]:
        """Return tensors transposed to the ``(B, T, ...)`` layout expected
        by :func:`train_dgppo.compute_dec_ocp_gae` /
        :func:`compute_cbf_advantages`.
        """
        return {
            "bT_l": -self.rewards.transpose(0, 1),                      # (B, T)
            "bTah_hs": self.costs.transpose(0, 1),                     # (B, T, A, NH)
            "bTp1_Vl": self.values_l.transpose(0, 1),                  # (B, T+1)
            "bTp1ah_Vh": self.values_h.transpose(0, 1),                # (B, T+1, A, NH)
            "bTah_Vh": self.values_h[: self.rollout_length].transpose(0, 1),
            "bT_Vl": self.values_l[: self.rollout_length].transpose(0, 1),
            "bTa_logp": self.log_prob.transpose(0, 1),                 # (B, T, A)
            "bTa_actions": self.actions.transpose(0, 1),               # (B, T, A, Da)
            "bTa_agent_state": self.agent_state.transpose(0, 1),       # (B, T, A, S)
            "bTa_goal_state": self.goal_state.transpose(0, 1),         # (B, T, A, S)
            "bTo_obs_state": self.obs_state.transpose(0, 1),           # (B, T, O, S)
        }

    def minibatch_iter(self, num_mini_batches: int):
        """Yield minibatches by splitting the ``B`` axis.

        DGPPO's PPO loop shuffles env indices and passes ``B/num_minibatches``
        trajectories per step, matching the JAX reference
        (``update_inner`` in ``dgppo/algo/dgppo.py``).
        """
        B = self.num_envs
        perm = torch.randperm(B, device=self.device)
        chunks = torch.chunk(perm, num_mini_batches)
        for idx in chunks:
            yield idx
