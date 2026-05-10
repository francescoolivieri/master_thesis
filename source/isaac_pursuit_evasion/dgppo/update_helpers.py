from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dgppo_models import DGPPOPolicy, DGPPOValueNet, TanhNormal
from .utils import (
    GraphData,
    build_graph_data,
    compute_policy_surrogate,
    graph_data_select,
    graph_data_slice,
)


@dataclass(frozen=True)
class UpdateGraphBatch:
    """Flattened minibatch view used by the DG-PPO update helpers."""

    graph: GraphData
    det_graph: GraphData
    actions: torch.Tensor
    old_logp: torch.Tensor
    advantages: torch.Tensor
    ql_targets: torch.Tensor
    qh_det_targets: torch.Tensor
    rnn_states: torch.Tensor | None
    vl_rnn_states: torch.Tensor | None
    det_rnn_states: torch.Tensor | None
    done_mask: torch.Tensor | None
    b: int
    T: int
    A: int

    @property
    def n_agents_total(self) -> int:
        return self.b * self.T * self.A

    @property
    def chunk_ids(self) -> torch.Tensor:
        return torch.arange(self.T, device=self.actions.device).reshape(1, self.T)


def build_update_graph_batch(
    *,
    idx: torch.Tensor,
    view: dict[str, torch.Tensor],
    det_view: dict[str, torch.Tensor],
    qh_det: torch.Tensor,
    ql: torch.Tensor,
    advantages: torch.Tensor,
    obs_radius: float,
    graph: GraphData | None = None,
    det_graph: GraphData | None = None,
) -> UpdateGraphBatch:
    """Gather one env-minibatch and build/select stochastic/deterministic graphs."""
    agent_s = view["bTa_agent_state"][idx]
    goal_s = view["bTa_goal_state"][idx]
    obs_s = view["bTo_obs_state"][idx]
    actions = view["bTa_actions"][idx]

    b, T, A, _ = actions.shape

    if graph is None:
        graph = build_rollout_graph(
            view={"bTa_agent_state": agent_s, "bTa_goal_state": goal_s, "bTo_obs_state": obs_s}, obs_radius=obs_radius
        )
    else:
        graph = select_rollout_envs(graph, idx=idx, T=T)

    if det_graph is None:
        det_agent_s = det_view["bTa_agent_state"][idx]
        det_goal_s = det_view["bTa_goal_state"][idx]
        det_obs_s = det_view["bTo_obs_state"][idx]
        det_graph = build_rollout_graph(
            view={"bTa_agent_state": det_agent_s, "bTa_goal_state": det_goal_s, "bTo_obs_state": det_obs_s},
            obs_radius=obs_radius,
        )
    else:
        det_graph = select_rollout_envs(det_graph, idx=idx, T=T)

    rnn_states = view.get("bTa_rnn_states")
    vl_rnn_states = view.get("bT_vl_rnn_states")
    det_rnn_states = det_view.get("bTa_rnn_states")
    done_mask = view.get("bT_done")

    return UpdateGraphBatch(
        graph=graph,
        det_graph=det_graph,
        actions=actions,
        old_logp=view["bTa_logp"][idx],
        advantages=advantages[idx],
        ql_targets=ql[idx],
        qh_det_targets=qh_det[idx],
        rnn_states=rnn_states[idx] if rnn_states is not None else None,
        vl_rnn_states=vl_rnn_states[idx] if vl_rnn_states is not None else None,
        det_rnn_states=det_rnn_states[idx] if det_rnn_states is not None else None,
        done_mask=done_mask[idx] if done_mask is not None else None,
        b=b,
        T=T,
        A=A,
    )


def build_rollout_graph(*, view: dict[str, torch.Tensor], obs_radius: float) -> GraphData:
    """Build a graph batch from a ``[B, T, ...]`` rollout view."""
    agent_s = view["bTa_agent_state"]
    goal_s = view["bTa_goal_state"]
    obs_s = view["bTo_obs_state"]
    B, T, A, _ = agent_s.shape
    BT = B * T
    return build_graph_data(
        agent_state=agent_s.reshape(BT, A, -1),
        goal_state=goal_s.reshape(BT, A, -1),
        obs_state=obs_s.reshape(BT, obs_s.shape[2], obs_s.shape[3]),
        obs_radius=obs_radius,
    )


def select_rollout_envs(graph: GraphData, *, idx: torch.Tensor, T: int) -> GraphData:
    """Select full trajectories from a flat ``[B*T]`` rollout graph batch."""
    time_ids = torch.arange(T, device=idx.device, dtype=torch.long)
    flat_ids = idx.long()[:, None] * T + time_ids[None, :]
    return graph_data_select(graph, flat_ids.reshape(-1))


def compute_policy_loss(
    *,
    policy: DGPPOPolicy,
    graph: GraphData,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    entropy_scale: float,
    n_agents_total: int,
    rnn_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the real policy and compute the clipped DG-PPO policy loss."""
    log_prob, entropy, _ = policy.evaluate(
        graph,
        action=actions.reshape(n_agents_total, -1),
        rnn_state=rnn_state,
        n_agents_total=n_agents_total,
        compute_entropy=entropy_scale > 0,
    )
    log_prob = log_prob.reshape_as(old_logp)
    loss_info = compute_policy_loss_from_log_prob(
        log_prob=log_prob,
        old_logp=old_logp,
        advantages=advantages,
        entropy=entropy,
        clip_eps=clip_eps,
        entropy_scale=entropy_scale,
    )

    return {
        **loss_info,
        "log_prob": log_prob,
        "entropy": entropy,
    }


def compute_policy_loss_from_log_prob(
    *,
    log_prob: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float,
    entropy_scale: float,
) -> dict[str, torch.Tensor]:
    """Compute DG-PPO's policy objective from already evaluated log-probs."""
    ratio = torch.exp(log_prob - old_logp.detach())
    surrogate = compute_policy_surrogate(ratio, advantages, clip_eps)
    entropy_mean = entropy.mean()
    entropy_bonus = -entropy_scale * entropy_mean if entropy_scale > 0 else surrogate["loss_policy"].new_zeros(())
    loss_total = surrogate["loss_policy"] + entropy_bonus

    return {
        **surrogate,
        "entropy_mean": entropy_mean,
        "entropy_bonus": entropy_bonus,
        "loss_policy_total": loss_total,
        "ratio": ratio,
    }


def compute_rollout_policy_loss(
    *,
    policy: DGPPOPolicy,
    graph: GraphData,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    chunk_ids: torch.Tensor,
    clip_eps: float,
    entropy_scale: float,
    n_agents: int,
    chunk_graph: GraphData | None = None,
    rnn_states: torch.Tensor | None = None,
    done_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate policy loss over rollout envs and RNN chunks using production policy code.

    If ``rnn_states`` is provided, it is expected as incoming per-step policy
    carries shaped ``[B, T, L, A, C, H]``. Chunk evaluation starts from the
    stored carry at each chunk's first timestep.
    """
    B, T, A, action_dim = actions.shape
    chunk_ids_index = _canonical_chunk_ids(chunk_ids, device=actions.device)
    C, R = chunk_ids_index.shape
    compute_entropy = entropy_scale > 0

    if chunk_graph is None:
        chunk_graph = rollout_graph_chunks(graph, chunk_ids=chunk_ids_index, T=T, B=B)

    action_chunks = actions[:, chunk_ids_index]
    if policy.rnn is None:
        step_log_prob, step_entropy, _ = policy.evaluate(
            chunk_graph,
            action=action_chunks.reshape(B * C * R * A, action_dim),
            rnn_state=None,
            n_agents_total=n_agents,
            compute_entropy=compute_entropy,
        )
        log_prob = step_log_prob.reshape(B, C, R, A)
        entropy = step_entropy.reshape(B, C, R, A)
    else:
        log_prob, entropy = _evaluate_policy_chunks(
            policy=policy,
            graph=chunk_graph,
            action_chunks=action_chunks,
            rnn_states=rnn_states[:, chunk_ids_index[:, 0]] if rnn_states is not None else None,
            done_chunks=done_mask[:, chunk_ids_index] if done_mask is not None else None,
            B=B,
            C=C,
            R=R,
            A=A,
            compute_entropy=compute_entropy,
        )

    old_logp_chunked = old_logp[:, chunk_ids_index]
    advantages_chunked = advantages[:, chunk_ids_index]
    loss_info = compute_policy_loss_from_log_prob(
        log_prob=log_prob,
        old_logp=old_logp_chunked,
        advantages=advantages_chunked,
        entropy=entropy,
        clip_eps=clip_eps,
        entropy_scale=entropy_scale,
    )
    return {
        **loss_info,
        "log_prob": log_prob,
        "entropy": entropy,
    }


def scan_policy_rnn_states(
    *,
    policy: DGPPOPolicy,
    graph: GraphData,
    B: int,
    T: int,
    A: int,
) -> torch.Tensor:
    """Scan the policy over rollout graphs and return incoming RNN states."""
    if policy.rnn is None:
        raise ValueError("scan_policy_rnn_states requires a recurrent policy")

    init_state = policy.initialize_carry(B * A, device=graph.nodes.device)
    states = graph.nodes.new_empty((B, T) + tuple(policy.initialize_carry(A, device=graph.nodes.device).shape))
    rnn_state = init_state
    for t in range(T):
        states[:, t] = rnn_state.reshape(rnn_state.shape[0], B, A, rnn_state.shape[2], rnn_state.shape[3]).permute(
            1, 0, 2, 3, 4
        )
        _, rnn_state = policy.distribution(rollout_graph_timestep(graph, t=t, T=T, B=B), rnn_state, A)
    return states


def rollout_graph_slice(graph: GraphData, b: int, t: int, *, T: int) -> GraphData:
    """Slice one ``(env, time)`` graph from either flat ``[B*T]`` or shaped ``[B, T]`` batches."""
    if graph.n_nodes.ndim == 1:
        return graph_data_slice(graph, b * T + t)
    return graph_data_slice(graph, (b, t))


def rollout_graph_timestep(graph: GraphData, *, t: int, T: int, B: int) -> GraphData:
    """Select all env graphs at one rollout timestep as a batched graph."""
    flat_ids = torch.arange(B, device=graph.nodes.device, dtype=torch.long) * T + int(t)
    return graph_data_select(graph, flat_ids)


def rollout_graph_chunks(graph: GraphData, *, chunk_ids: torch.Tensor, T: int, B: int) -> GraphData:
    """Select rollout graphs as ``[B, C, R]`` chunk batches.

    ``graph`` is stored env-major as ``[B*T]``. The returned graph keeps the
    truncated-BPTT chunk structure explicit so GNN/MLP work can be batched while
    the recurrent state still resets once per chunk.
    """
    chunk_ids_index = _canonical_chunk_ids(chunk_ids, device=graph.nodes.device)
    env_ids = torch.arange(B, device=graph.nodes.device, dtype=torch.long).reshape(B, 1, 1)
    flat_ids = env_ids * T + chunk_ids_index.reshape(1, *chunk_ids_index.shape)
    return graph_data_select(graph, flat_ids)


def compute_value_losses(
    *,
    Vl: DGPPOValueNet,
    Vh: DGPPOValueNet,
    graph: GraphData,
    det_graph: GraphData,
    ql_targets: torch.Tensor,
    qh_det_targets: torch.Tensor,
    A: int,
    vl_loss_scale: float,
    vh_loss_scale: float,
    rnn_states: torch.Tensor | None = None,
    vl_rnn_states: torch.Tensor | None = None,
    det_rnn_states: torch.Tensor | None = None,
    done_mask: torch.Tensor | None = None,
    chunk_ids: torch.Tensor | None = None,
    chunk_graph: GraphData | None = None,
    det_chunk_graph: GraphData | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the real low-level and safety critic losses for one minibatch."""
    b, T = ql_targets.shape

    if Vl.rnn is not None or Vh.rnn is not None:
        if chunk_ids is None:
            chunk_ids = torch.arange(T, device=ql_targets.device).reshape(1, T)
        vl_info = compute_rollout_vl_loss(
            Vl=Vl,
            graph=graph,
            targets=ql_targets,
            chunk_ids=chunk_ids,
            A=A,
            loss_scale=vl_loss_scale,
            rnn_states=vl_rnn_states,
            done_mask=done_mask,
            chunk_graph=chunk_graph,
        )
        if det_rnn_states is None:
            raise ValueError("Recurrent Vh update requires deterministic rollout RNN states")
        vh_info = compute_rollout_vh_loss(
            Vh=Vh,
            graph=det_graph,
            rnn_states=det_rnn_states,
            targets=qh_det_targets,
            chunk_ids=chunk_ids,
            A=A,
            loss_scale=vh_loss_scale,
            chunk_graph=det_chunk_graph,
        )
        return {
            "vl": vl_info["vl"],
            "vh": vh_info["vh"],
            "loss_vl": vl_info["loss_vl"],
            "loss_vh": vh_info["loss_vh"],
        }

    vl, _ = Vl(graph, None, A)
    vl = vl.reshape(b, T)
    loss_vl = compute_value_l2_loss(vl, ql_targets, scale=vl_loss_scale)

    vh, _ = Vh(det_graph, None, A)
    vh = vh.reshape(b, T, A, -1)
    loss_vh = compute_value_l2_loss(vh, qh_det_targets, scale=vh_loss_scale)

    return {
        "vl": vl,
        "vh": vh,
        "loss_vl": loss_vl,
        "loss_vh": loss_vh,
    }


def compute_value_l2_loss(prediction: torch.Tensor, target: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Match Optax ``l2_loss(...).mean()``: ``0.5 * squared_error.mean()``."""
    return scale * 0.5 * F.mse_loss(prediction, target)


def compute_rollout_vl_loss(
    *,
    Vl: DGPPOValueNet,
    graph: GraphData,
    targets: torch.Tensor,
    chunk_ids: torch.Tensor,
    A: int,
    loss_scale: float = 1.0,
    rnn_states: torch.Tensor | None = None,
    done_mask: torch.Tensor | None = None,
    chunk_graph: GraphData | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the Vl update loss over rollout chunks using stored chunk-start RNN states when available."""
    B, T = targets.shape
    chunk_ids_index = _canonical_chunk_ids(chunk_ids, device=targets.device)
    C, R = chunk_ids_index.shape
    if chunk_graph is None:
        chunk_graph = rollout_graph_chunks(graph, chunk_ids=chunk_ids_index, T=T, B=B)
    values = _evaluate_vl_chunks(
        Vl=Vl,
        graph=chunk_graph,
        rnn_states=rnn_states[:, chunk_ids_index[:, 0]] if rnn_states is not None else None,
        done_chunks=done_mask[:, chunk_ids_index] if done_mask is not None else None,
        B=B,
        C=C,
        R=R,
        A=A,
        device=targets.device,
    )
    targets_chunked = targets[:, chunk_ids_index]
    return {
        "vl": values,
        "loss_vl": compute_value_l2_loss(values, targets_chunked, scale=loss_scale),
    }


def compute_rollout_vh_loss(
    *,
    Vh: DGPPOValueNet,
    graph: GraphData,
    rnn_states: torch.Tensor,
    targets: torch.Tensor,
    chunk_ids: torch.Tensor,
    A: int,
    loss_scale: float = 1.0,
    chunk_graph: GraphData | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the deterministic Vh update loss over rollout chunks."""
    B, T, _A, n_cost = targets.shape
    chunk_ids_index = _canonical_chunk_ids(chunk_ids, device=targets.device)
    C, R = chunk_ids_index.shape
    if chunk_graph is None:
        chunk_graph = rollout_graph_chunks(graph, chunk_ids=chunk_ids_index, T=T, B=B)
    values = _evaluate_vh_chunks(
        Vh=Vh,
        graph=chunk_graph,
        rnn_states=rnn_states[:, chunk_ids_index],
        B=B,
        C=C,
        R=R,
        A=A,
        n_cost=n_cost,
    )
    targets_chunked = targets[:, chunk_ids_index]
    return {
        "vh": values,
        "loss_vh": compute_value_l2_loss(values, targets_chunked, scale=loss_scale),
    }


def scan_vl_values(
    *,
    Vl: DGPPOValueNet,
    graph: GraphData,
    B: int,
    T: int,
    A: int,
) -> torch.Tensor:
    """Run the centralized value scan over each rollout environment."""
    values = graph.nodes.new_empty((B, T))
    rnn_state = Vl.rnn.initialize_carry(B, device=graph.nodes.device) if Vl.rnn is not None else None
    for t in range(T):
        value, rnn_state = Vl(rollout_graph_timestep(graph, t=t, T=T, B=B), rnn_state, A)
        values[:, t] = value.squeeze(-1)
    return values


def evaluate_vh_values(
    *,
    Vh: DGPPOValueNet,
    graph: GraphData,
    rnn_states: torch.Tensor,
    B: int,
    T: int,
    A: int,
) -> torch.Tensor:
    """Evaluate decomposed safety values using the rollout policy RNN states."""
    first_state = rnn_states[0, 0]
    n_heads = Vh.net.n_out
    values = graph.nodes.new_empty((B, T, A, n_heads), dtype=first_state.dtype)
    for t in range(T):
        step_rnn_states = _flatten_agent_rnn_states(rnn_states[:, t])
        value, _ = Vh(rollout_graph_timestep(graph, t=t, T=T, B=B), step_rnn_states, A)
        values[:, t] = value
    return values


def _flatten_agent_rnn_states(rnn_states: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., L, A, C, H]`` states to model carry ``[L, prod(...)*A, C, H]``."""
    if rnn_states.ndim < 5:
        raise ValueError(f"expected at least 5 RNN-state dims, got shape {tuple(rnn_states.shape)}")
    prefix_shape = tuple(int(size) for size in rnn_states.shape[:-4])
    L, A, n_carries, H = (int(size) for size in rnn_states.shape[-4:])
    n_prefix = 1
    for size in prefix_shape:
        n_prefix *= size
    prefix_ndim = len(prefix_shape)
    permute_order = (
        prefix_ndim,
        *range(prefix_ndim),
        prefix_ndim + 1,
        prefix_ndim + 2,
        prefix_ndim + 3,
    )
    return rnn_states.permute(permute_order).reshape(L, n_prefix * A, n_carries, H)


def _canonical_chunk_ids(chunk_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    chunk_ids_index = chunk_ids.to(device=device, dtype=torch.long)
    if chunk_ids_index.ndim == 1:
        return chunk_ids_index.reshape(1, -1)
    if chunk_ids_index.ndim != 2:
        raise ValueError(f"chunk_ids must have shape [C, R] or [R], got {tuple(chunk_ids_index.shape)}")
    return chunk_ids_index


def _evaluate_policy_chunks(
    *,
    policy: DGPPOPolicy,
    graph: GraphData,
    action_chunks: torch.Tensor,
    rnn_states: torch.Tensor | None,
    done_chunks: torch.Tensor | None,
    B: int,
    C: int,
    R: int,
    A: int,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate recurrent policy chunks with one GNN/MLP batch and an R-step recurrent scan."""
    action_dim = action_chunks.shape[-1]
    x = policy.gnn(graph, node_type=0, n_type=A)
    x = policy.mlp(x).reshape(B, C, R, A, policy.mlp.hid_sizes[-1])
    x = x.permute(2, 0, 1, 3, 4).reshape(R, B * C * A, policy.mlp.hid_sizes[-1])

    if rnn_states is None:
        rnn_state = policy.initialize_carry(B * C * A, device=action_chunks.device)
    else:
        rnn_state = _flatten_agent_rnn_states(rnn_states)
    log_prob = action_chunks.new_empty((B, C, R, A))
    entropy = action_chunks.new_zeros((B, C, R, A))

    for r in range(R):
        x_r, rnn_state = policy.rnn(x[r], rnn_state)
        h = policy.scale_hid(x_r)
        mean = policy.mean_head(h).reshape(B, C, A, action_dim)
        std = F.softplus(policy.std_head(h) + policy.std_dev_init_inv) + policy.std_dev_min
        std = std.reshape(B, C, A, action_dim)
        dist = TanhNormal(mean, std)
        log_prob[:, :, r] = dist.log_prob(action_chunks[:, :, r])
        if compute_entropy:
            entropy[:, :, r] = dist.entropy()
        if done_chunks is not None:
            rnn_state = _reset_policy_carry_after_done(rnn_state, done_chunks[:, :, r], n_agents=A)

    return log_prob, entropy


def _evaluate_vl_chunks(
    *,
    Vl: DGPPOValueNet,
    graph: GraphData,
    rnn_states: torch.Tensor | None,
    done_chunks: torch.Tensor | None,
    B: int,
    C: int,
    R: int,
    A: int,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate centralized Vl chunks while preserving per-chunk recurrent resets."""
    if Vl.rnn is None:
        values, _ = Vl(graph, None, A)
        return values.squeeze(-1)

    x = Vl.gnn(graph, node_type=0, n_type=A)
    x = x.mean(dim=-2)
    x = Vl.head(x).reshape(B, C, R, Vl.head.hid_sizes[-1])
    x = x.permute(2, 0, 1, 3).reshape(R, B * C, Vl.head.hid_sizes[-1])

    if rnn_states is None:
        rnn_state = Vl.rnn.initialize_carry(B * C, device=device)
    else:
        rnn_state = _flatten_env_rnn_states(rnn_states)
    values = x.new_empty((B, C, R, Vl.net.n_out))
    for r in range(R):
        x_r, rnn_state = Vl.rnn(x[r], rnn_state)
        values[:, :, r] = Vl.net.value_out(x_r).reshape(B, C, Vl.net.n_out)
        if done_chunks is not None:
            rnn_state = _reset_env_carry_after_done(rnn_state, done_chunks[:, :, r])
    return values.squeeze(-1)


def _reset_policy_carry_after_done(
    rnn_state: torch.Tensor,
    done: torch.Tensor,
    *,
    n_agents: int,
) -> torch.Tensor:
    """Zero finished env slots in a policy carry without mutating autograd state."""
    done_1d = torch.as_tensor(done, device=rnn_state.device, dtype=torch.bool).reshape(-1)
    if done_1d.numel() == 0 or not bool(done_1d.any().item()):
        return rnn_state
    L, total_agents, n_carries, H = rnn_state.shape
    n_agents = int(n_agents)
    if n_agents <= 0 or total_agents % n_agents != 0:
        raise ValueError(f"cannot reshape policy RNN state with total_agents={total_agents}, n_agents={n_agents}")
    n_envs = total_agents // n_agents
    if done_1d.numel() != n_envs:
        raise ValueError(f"done mask has {done_1d.numel()} envs, but policy RNN state has {n_envs}")
    keep = (~done_1d).to(dtype=rnn_state.dtype).reshape(1, n_envs, 1, 1, 1)
    return (rnn_state.reshape(L, n_envs, n_agents, n_carries, H) * keep).reshape_as(rnn_state)


def _reset_env_carry_after_done(rnn_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """Zero finished env slots in a centralized carry without mutating autograd state."""
    done_1d = torch.as_tensor(done, device=rnn_state.device, dtype=torch.bool).reshape(-1)
    if done_1d.numel() == 0 or not bool(done_1d.any().item()):
        return rnn_state
    if rnn_state.shape[1] != done_1d.numel():
        raise ValueError(f"done mask has {done_1d.numel()} envs, but RNN state has {rnn_state.shape[1]}")
    keep = (~done_1d).to(dtype=rnn_state.dtype).reshape(1, -1, 1, 1)
    return rnn_state * keep


def _flatten_env_rnn_states(rnn_states: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., L, C, H]`` states to model carry ``[L, prod(...), C, H]``."""
    if rnn_states.ndim < 4:
        raise ValueError(f"expected at least 4 RNN-state dims, got shape {tuple(rnn_states.shape)}")
    prefix_shape = tuple(int(size) for size in rnn_states.shape[:-3])
    L, n_carries, H = (int(size) for size in rnn_states.shape[-3:])
    n_prefix = 1
    for size in prefix_shape:
        n_prefix *= size
    prefix_ndim = len(prefix_shape)
    permute_order = (prefix_ndim, *range(prefix_ndim), prefix_ndim + 1, prefix_ndim + 2)
    return rnn_states.permute(permute_order).reshape(L, n_prefix, n_carries, H)


def _evaluate_vh_chunks(
    *,
    Vh: DGPPOValueNet,
    graph: GraphData,
    rnn_states: torch.Tensor,
    B: int,
    C: int,
    R: int,
    A: int,
    n_cost: int,
) -> torch.Tensor:
    """Evaluate deterministic Vh chunks in one batched call using stored policy RNN states."""
    rnn_state = _flatten_agent_rnn_states(rnn_states) if Vh.rnn is not None else None
    values, _ = Vh(graph, rnn_state, A)
    return values.reshape(B, C, R, A, n_cost)


def apply_policy_update(
    *,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    grad_clip: float,
) -> torch.Tensor:
    """Backpropagate, clip gradients, and apply one optimizer step."""
    return _apply_optimizer_update(optimizer=optimizer, loss=loss, parameters=parameters, grad_clip=grad_clip)


def apply_value_update(
    *,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    grad_clip: float,
) -> torch.Tensor:
    """Backpropagate, clip gradients, and apply one critic optimizer step."""
    return _apply_optimizer_update(optimizer=optimizer, loss=loss, parameters=parameters, grad_clip=grad_clip)


def _apply_optimizer_update(
    *,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    grad_clip: float,
) -> torch.Tensor:
    params = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    return torch.as_tensor(grad_norm, device=loss.device, dtype=loss.dtype)
