"""Task adapters for turning IsaacLab env state into DGPPO inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .gnn import GraphData


@dataclass
class DGPPOAdapterOutput:
    """Per-step data extracted from an IsaacLab task."""

    graph: GraphData[torch.Tensor]
    constraint_costs: torch.Tensor  # [B, A, NH], currently A=1 for PosTracking
    reward_cost: torch.Tensor  # [B], usually -reward


class TaskAdapter(Protocol):
    def extract(self, env, reward: torch.Tensor | None = None) -> DGPPOAdapterOutput:
        """Build DGPPO-ready tensors from the current environment state."""


class PosTrackingGraphAdapter:
    """
    Adapter from `PosTrackingEnv` state to DGPPO graph/cost tensors.

    The graph keeps a fixed per-env structure:
        [agent, goal, obstacle_0..obstacle_n-1, padding]
    """

    AGENT_NODE_TYPE = 0
    GOAL_NODE_TYPE = 1
    OBSTACLE_NODE_TYPE = 2
    PAD_NODE_TYPE = -1

    def __init__(
        self,
        obstacle_clearance: float = 0.0,
        n_constraint_heads: int = 3,
    ) -> None:
        self.obstacle_clearance = float(obstacle_clearance)
        self.n_constraint_heads = int(n_constraint_heads)

    def extract(self, env, reward: torch.Tensor | None = None) -> DGPPOAdapterOutput:
        device = env.device
        env_origins = env._terrain.env_origins
        pos_local = env._robot.data.root_pos_w - env_origins
        vel_world = env._robot.data.root_lin_vel_w
        ang_vel = env._robot.data.root_ang_vel_b
        quat = env._robot.data.root_quat_w

        pos_error = env._reference_pos - pos_local
        dist = torch.norm(pos_error, dim=-1, keepdim=True)
        agent_feats = torch.cat([pos_error, dist, vel_world, ang_vel, quat], dim=-1)  # [B, 14]
        feat_dim = int(agent_feats.shape[-1])

        goal_delta = env._reference_pos - pos_local
        goal_dist = torch.norm(goal_delta, dim=-1, keepdim=True)
        goal_feats = torch.cat(
            [
                goal_delta,
                goal_dist,
                torch.zeros(env.num_envs, feat_dim - 4, device=device, dtype=agent_feats.dtype),
            ],
            dim=-1,
        )  # [B, 14]

        num_obs = int(getattr(env, "_num_pillars", 0))
        if num_obs > 0:
            rel_xy = env._pillar_positions_xy.unsqueeze(0) - pos_local[:, :2].unsqueeze(1)  # [B, O, 2]
            center_dist = torch.norm(rel_xy, dim=-1, keepdim=True)  # [B, O, 1]
            surface_dist = center_dist - env._pillar_collision_radius
            obs_core = torch.cat([rel_xy, surface_dist], dim=-1)  # [B, O, 3]
            obs_pad = torch.zeros(env.num_envs, num_obs, feat_dim - 3, device=device, dtype=agent_feats.dtype)
            obs_feats = torch.cat([obs_core, obs_pad], dim=-1)  # [B, O, 14]
        else:
            obs_feats = torch.zeros(env.num_envs, 0, feat_dim, device=device, dtype=agent_feats.dtype)

        pad_feats = torch.zeros(env.num_envs, 1, feat_dim, device=device, dtype=agent_feats.dtype)
        nodes_per_graph = 1 + 1 + num_obs + 1
        nodes = torch.cat(
            [
                agent_feats.unsqueeze(1),
                goal_feats.unsqueeze(1),
                obs_feats,
                pad_feats,
            ],
            dim=1,
        )  # [B, N, F]

        node_types = torch.cat(
            [
                torch.full((env.num_envs, 1), self.AGENT_NODE_TYPE, dtype=torch.long, device=device),
                torch.full((env.num_envs, 1), self.GOAL_NODE_TYPE, dtype=torch.long, device=device),
                torch.full((env.num_envs, num_obs), self.OBSTACLE_NODE_TYPE, dtype=torch.long, device=device),
                torch.full((env.num_envs, 1), self.PAD_NODE_TYPE, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        edges_per_graph = (nodes_per_graph - 1) * (nodes_per_graph - 2)
        senders_per_graph = []
        receivers_per_graph = []
        edge_feats_per_graph = []
        valid_ids = torch.arange(nodes_per_graph - 1, device=device, dtype=torch.long)
        for recv in valid_ids.tolist():
            for send in valid_ids.tolist():
                if recv == send:
                    continue
                receivers_per_graph.append(recv)
                senders_per_graph.append(send)

        local_receivers = torch.tensor(receivers_per_graph, device=device, dtype=torch.long)
        local_senders = torch.tensor(senders_per_graph, device=device, dtype=torch.long)

        flat_nodes = nodes.reshape(env.num_envs * nodes_per_graph, feat_dim)
        flat_types = node_types.reshape(env.num_envs * nodes_per_graph)
        n_nodes = torch.full((env.num_envs,), nodes_per_graph, device=device, dtype=torch.long)
        n_edges = torch.full((env.num_envs,), edges_per_graph, device=device, dtype=torch.long)

        graph_receivers = []
        graph_senders = []
        graph_edges = []
        for b in range(env.num_envs):
            offset = b * nodes_per_graph
            rec = local_receivers + offset
            snd = local_senders + offset
            graph_receivers.append(rec)
            graph_senders.append(snd)
            graph_edges.append(flat_nodes[rec] - flat_nodes[snd])

        receivers = torch.cat(graph_receivers, dim=0)
        senders = torch.cat(graph_senders, dim=0)
        edges = torch.cat(graph_edges, dim=0)

        graph = GraphData(
            n_nodes=n_nodes,
            n_edges=n_edges,
            nodes=flat_nodes,
            edges=edges,
            states=flat_nodes,
            receivers=receivers,
            senders=senders,
            node_types=flat_types,
        )

        # Constraint costs for current state:
        #   h0: below min safe altitude
        #   h1: outside safe arena bounds
        #   h2: obstacle penetration
        z = pos_local[:, 2]
        h_alt = torch.relu(env._arena_min_safe[2] - z)

        below_min = torch.relu(env._arena_min_safe - pos_local)
        above_max = torch.relu(pos_local - env._arena_max_safe)
        h_bounds = (below_min + above_max).sum(dim=-1)

        if num_obs > 0:
            rel_xy = pos_local[:, None, :2] - env._pillar_positions_xy.unsqueeze(0)
            d_center = torch.norm(rel_xy, dim=-1)
            safety_radius = env._pillar_collision_radius + self.obstacle_clearance
            h_obs = torch.relu(safety_radius - d_center).max(dim=1).values
        else:
            h_obs = torch.zeros_like(h_alt)

        heads = [h_alt, h_bounds, h_obs]
        if self.n_constraint_heads > len(heads):
            for _ in range(self.n_constraint_heads - len(heads)):
                heads.append(torch.zeros_like(h_alt))
        cost_heads = torch.stack(heads[: self.n_constraint_heads], dim=-1)
        constraint_costs = cost_heads.unsqueeze(1)  # [B, 1, NH]

        reward_cost = -reward if reward is not None else torch.zeros(env.num_envs, device=device)
        return DGPPOAdapterOutput(
            graph=graph,
            constraint_costs=constraint_costs,
            reward_cost=reward_cost,
        )
