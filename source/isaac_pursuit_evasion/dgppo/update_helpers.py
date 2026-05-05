from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .dgppo_models import DGPPOPolicy, DGPPOValueNet
from .utils import GraphData, build_graph_data, compute_policy_surrogate, graph_data_slice


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
    rnn_states: Optional[torch.Tensor]
    det_rnn_states: Optional[torch.Tensor]
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
) -> UpdateGraphBatch:
    """Gather one env-minibatch and rebuild the stochastic/deterministic graphs."""
    agent_s = view["bTa_agent_state"][idx]
    goal_s = view["bTa_goal_state"][idx]
    obs_s = view["bTo_obs_state"][idx]
    actions = view["bTa_actions"][idx]

    b, T, A, _ = actions.shape
    BT = b * T

    graph = build_graph_data(
        agent_state=agent_s.reshape(BT, A, -1),
        goal_state=goal_s.reshape(BT, A, -1),
        obs_state=obs_s.reshape(BT, obs_s.shape[2], obs_s.shape[3]),
        obs_radius=obs_radius,
    )
    det_agent_s = det_view["bTa_agent_state"][idx]
    det_goal_s = det_view["bTa_goal_state"][idx]
    det_obs_s = det_view["bTo_obs_state"][idx]
    det_graph = build_graph_data(
        agent_state=det_agent_s.reshape(BT, A, -1),
        goal_state=det_goal_s.reshape(BT, A, -1),
        obs_state=det_obs_s.reshape(BT, det_obs_s.shape[2], det_obs_s.shape[3]),
        obs_radius=obs_radius,
    )
    rnn_states = view.get("bTa_rnn_states", None)
    det_rnn_states = det_view.get("bTa_rnn_states", None)

    return UpdateGraphBatch(
        graph=graph,
        det_graph=det_graph,
        actions=actions,
        old_logp=view["bTa_logp"][idx],
        advantages=advantages[idx],
        ql_targets=ql[idx],
        qh_det_targets=qh_det[idx],
        rnn_states=rnn_states[idx] if rnn_states is not None else None,
        det_rnn_states=det_rnn_states[idx] if det_rnn_states is not None else None,
        b=b,
        T=T,
        A=A,
    )


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
    rnn_state: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the real policy and compute the clipped DG-PPO policy loss."""
    log_prob, entropy, _ = policy.evaluate(
        graph,
        action=actions.reshape(n_agents_total, -1),
        rnn_state=rnn_state,
        n_agents_total=n_agents_total,
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
) -> dict[str, torch.Tensor]:
    """Evaluate policy loss over rollout envs and RNN chunks using production policy code."""
    B, _T, A, action_dim = actions.shape
    C, R = chunk_ids.shape
    log_prob = old_logp.new_empty((B, C, R, A))
    entropy = old_logp.new_empty((B, C, R, A))

    for b in range(B):
        for c in range(C):
            rnn_state = policy.initialize_carry(n_agents, device=actions.device) if policy.rnn is not None else None
            for r in range(R):
                t = int(chunk_ids[c, r].item())
                step_graph = rollout_graph_slice(graph, b, t, T=actions.shape[1])
                step_action = actions[b, t].reshape(A, action_dim)
                step_log_prob, step_entropy, rnn_state = policy.evaluate(
                    step_graph,
                    action=step_action,
                    rnn_state=rnn_state,
                    n_agents_total=n_agents,
                )
                log_prob[b, c, r] = step_log_prob
                entropy[b, c, r] = step_entropy

    old_logp_chunked = old_logp[:, chunk_ids.long()]
    advantages_chunked = advantages[:, chunk_ids.long()]
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

    init_state = policy.initialize_carry(A, device=graph.nodes.device)
    states = graph.nodes.new_empty((B, T) + tuple(init_state.shape))
    for b in range(B):
        rnn_state = policy.initialize_carry(A, device=graph.nodes.device)
        for t in range(T):
            states[b, t] = rnn_state
            _, rnn_state = policy.distribution(graph_data_slice(graph, (b, t)), rnn_state, A)
    return states


def rollout_graph_slice(graph: GraphData, b: int, t: int, *, T: int) -> GraphData:
    """Slice one ``(env, time)`` graph from either flat ``[B*T]`` or shaped ``[B, T]`` batches."""
    if graph.n_nodes.ndim == 1:
        return graph_data_slice(graph, b * T + t)
    return graph_data_slice(graph, (b, t))


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
    rnn_states: Optional[torch.Tensor] = None,
    det_rnn_states: Optional[torch.Tensor] = None,
    chunk_ids: Optional[torch.Tensor] = None,
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
) -> dict[str, torch.Tensor]:
    """Compute the Vl update loss over rollout chunks using zero RNN chunk starts."""
    B, _T = targets.shape
    C, R = chunk_ids.shape
    values = targets.new_empty((B, C, R))
    for b in range(B):
        for c in range(C):
            rnn_state = Vl.rnn.initialize_carry(1, device=targets.device) if Vl.rnn is not None else None
            for r in range(R):
                t = int(chunk_ids[c, r].item())
                value, rnn_state = Vl(rollout_graph_slice(graph, b, t, T=targets.shape[1]), rnn_state, A)
                values[b, c, r] = value.squeeze(0).squeeze(-1)
    targets_chunked = targets[:, chunk_ids.long()]
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
) -> dict[str, torch.Tensor]:
    """Compute the deterministic Vh update loss over rollout chunks."""
    B, _T, _A, n_cost = targets.shape
    C, R = chunk_ids.shape
    values = targets.new_empty((B, C, R, A, n_cost))
    for b in range(B):
        for c in range(C):
            for r in range(R):
                t = int(chunk_ids[c, r].item())
                value, _ = Vh(rollout_graph_slice(graph, b, t, T=targets.shape[1]), rnn_states[b, t], A)
                values[b, c, r] = value
    targets_chunked = targets[:, chunk_ids.long()]
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
    for b in range(B):
        rnn_state = Vl.rnn.initialize_carry(1, device=graph.nodes.device) if Vl.rnn is not None else None
        for t in range(T):
            value, rnn_state = Vl(graph_data_slice(graph, (b, t)), rnn_state, A)
            values[b, t] = value.squeeze(0).squeeze(-1)
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
    for b in range(B):
        for t in range(T):
            value, _ = Vh(graph_data_slice(graph, (b, t)), rnn_states[b, t], A)
            values[b, t] = value
    return values


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
