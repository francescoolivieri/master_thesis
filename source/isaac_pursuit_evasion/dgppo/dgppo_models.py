from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import GraphData, GraphTransformerGNN
from .utils import MLP, RNN



# ---------------------------------------------------------------------------
# Value networks
# ---------------------------------------------------------------------------


class DecStateFn(nn.Module):
    """
    Decentralized head: one output per agent. 
    
    Applies the shared GNN, filters to agent nodes, passes each agent through the shared MLP/RNN,
    and projects to ``n_out``.
    """

    def __init__(self, gnn: nn.Module, mlp: nn.Module, rnn: Optional[nn.Module] = None, n_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.rnn = rnn
        self.n_out = n_out

        # final projection 
        self.value_out = nn.Linear(mlp.hid_sizes[-1], n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        x = self.gnn(graph, node_type=0, n_type=n_agents)    # (n_agents, gnn_out_dim)
        batch_shape = x.shape[:-2]
        x = self.mlp(x).reshape(-1, self.mlp.hid_sizes[-1])   # (B*n_agents, hid)

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)                                 # (n_agents, n_out)
        x = x.reshape(batch_shape + (n_agents, self.n_out))
        return x, rnn_state


class RStateFn(nn.Module):
    """
    Centralized head: one output per sub-graph.
    
    Same as 'DecStateFn' but aggregates per-agent GNN outputs with a mean before the MLP/RNN.
    """

    def __init__(self, gnn: nn.Module, mlp: nn.Module, rnn: Optional[nn.Module] = None, n_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.rnn = rnn
        self.n_out = n_out

        self.value_out = nn.Linear(mlp.hid_sizes[-1], n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        x = self.gnn(graph, node_type=0, n_type=n_agents)    # (n_agents, gnn_out_dim)
        batch_shape = x.shape[:-2]
        x = x.mean(dim=-2)                                    # (..., gnn_out_dim)
        x = self.mlp(x).reshape(-1, self.mlp.hid_sizes[-1])

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)                                 # (1, n_out)
        x = x.reshape(batch_shape + (self.n_out,))
        return x, rnn_state

class DGPPOValueNet(nn.Module):
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        gnn_layers: int = 1,
        gnn_out_dim: int = 64,
        gnn_msg_dim: int = 32,
        gnn_heads: int = 3,
        mlp_hid: tuple[int, ...] = (64, 64),
        use_rnn: bool = True,
        rnn_cell: str = "gru",
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        n_out: int = 1,
        decompose: bool = False,
        device: torch.device | str | None = None,
    ) -> None:

        super().__init__()
        # skrl Agent.__init__ reads model.device to call model.to(model.device)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.use_rnn = use_rnn
        self.decompose = decompose

        # Note: same backbone as the policy net, different heads
        self.gnn = GraphTransformerGNN(
            in_dim=node_dim,
            edge_dim=edge_dim,
            msg_dim=gnn_msg_dim,
            out_dim=gnn_out_dim,
            n_heads=gnn_heads,
            n_layers=gnn_layers,
        )
        self.head = MLP(
            hid_sizes=mlp_hid,
            in_dim=gnn_out_dim,
            act=nn.functional.relu,
            act_final=True,
        )
        self.rnn = (
            RNN(
                rnn_cell=rnn_cell,
                input_size=mlp_hid[-1],
                hidden_size=rnn_hidden,
                rnn_layers=rnn_layers,
            )
            if use_rnn
            else None
        )

        head_cls = DecStateFn if decompose else RStateFn
        self.net = head_cls(gnn=self.gnn, mlp=self.head, rnn=self.rnn, n_out=n_out)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        # GNN -> MLP -> RNN (optional) -> Final projection
        return self.net(graph, rnn_state, n_agents)

    def enable_training_mode(self, enabled: bool = True) -> None:
        """Called by skrl Agent.enable_models_training_mode()."""
        self.train(enabled)
    

# ---------------------------------------------------------------------------
# Policy network
# --------------------------------------------------------------------------

from torch.distributions import Normal

class TanhNormal:
    """
    Squashed-Gaussian distribution used by the DGPPO policy.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, threshold: float = 0.999) -> None:
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)
        
        self.threshold = threshold
        self.inverse_threshold = math.atanh(threshold)
        
        # average(pdf) = p / epsilon -> log(average(pdf)) = log(p) - log(epsilon)
        self.log_epsilon = math.log(1.0 - threshold)

    def sample(self, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample with optional fixed standard-normal noise for parity tests."""
        if noise is None:
            u = self.normal.rsample()
        else:
            u = self.mean + self.std * noise
        return torch.tanh(u)
    
    def _log_cdf(self, x: float) -> torch.Tensor:
        """Numerically stable log CDF of the Normal distribution."""
        z = (x - self.mean) / self.std
        # Using identity: cdf(x) = 0.5 * erfc(-z / sqrt(2))
        erfc_val = torch.clamp(torch.special.erfc(-z / math.sqrt(2)), min=1e-8)
        return math.log(0.5) + torch.log(erfc_val)

    def _log_survival(self, x: float) -> torch.Tensor:
        """Numerically stable log Survival function (1 - CDF) of the Normal distribution."""
        z = (x - self.mean) / self.std
        # Using identity: survival(x) = 0.5 * erfc(z / sqrt(2))
        erfc_val = torch.clamp(torch.special.erfc(z / math.sqrt(2)), min=1e-8)
        return math.log(0.5) + torch.log(erfc_val)

    def log_prob(self, action: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Log-density of a squashed action under this distribution.
        """
        # Clip to avoid atanh(+/-1) -> inf.
        a_clipped = torch.clamp(action, -self.threshold, self.threshold)
        
        # Calculate inner log-prob
        u = torch.atanh(a_clipped)
        log_p_u = self.normal.log_prob(u)
        # Inverse log det Jacobian: -log(1 - y^2)
        log_det_inv = torch.log(torch.clamp(1.0 - a_clipped.pow(2), min=eps))
        inner_log_prob = log_p_u - log_det_inv
        
        # Calculate the left and right tail log-probs
        log_prob_left = self._log_cdf(-self.inverse_threshold) - self.log_epsilon
        log_prob_right = self._log_survival(self.inverse_threshold) - self.log_epsilon
        
        # Route the calculation based on the clipped boundaries
        log_prob_components = torch.where(
            a_clipped <= -self.threshold,
            log_prob_left,
            torch.where(
                a_clipped >= self.threshold, 
                log_prob_right, 
                inner_log_prob
            )
        )
        
        # Sum over the action dimensions
        return log_prob_components.sum(dim=-1)

    def entropy(self, n_samples: int = 1) -> torch.Tensor:
        """MC estimate of the differential entropy of the squashed Gaussian.

        Closed form does not exist; a single reparam sample is enough for
        the small regularization weight DGPPO uses (``coef_ent``).
        """
        # Exact analytical entropy of the underlying Normal distribution
        base_entropy = self.normal.entropy()
        
        # Single sample MC estimate of the forward log-det-Jacobian
        # Tanh Jacobian: log(1 - tanh(u)^2) evaluated on a sample from the base distribution
        u = self.normal.rsample()
        a = torch.tanh(u)
        log_det_jacobian = torch.log(torch.clamp(1.0 - a.pow(2), min=1e-8))
        
        # Combine and sum over the action dimensions
        return (base_entropy + log_det_jacobian).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)


class DGPPOPolicy(nn.Module):
    """
    Decentralized policy producing one TanhNormal per agent.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        action_dim: int,
        *,
        gnn_layers: int = 1,
        gnn_out_dim: int = 64,
        gnn_msg_dim: int = 32,
        gnn_heads: int = 3,
        mlp_hid: tuple[int, ...] = (64, 64),
        scale_hid: int = 64,
        scale_final: float = 0.01,
        std_dev_init: float = 0.5,
        std_dev_min: float = 1e-5,
        use_rnn: bool = False,
        rnn_cell: str = "gru",
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        device: torch.device | str | None = None,
    ) -> None:

        super().__init__()
        # skrl Agent.__init__ reads model.device to call model.to(model.device)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.std_dev_min = std_dev_min
        self.std_dev_init = std_dev_init
        self.std_dev_init_inv = math.log(math.exp(std_dev_init) - 1.0)

        # Note: same backbone as the value net, different heads
        self.use_rnn = use_rnn

        self.gnn = GraphTransformerGNN(
            in_dim=node_dim,
            edge_dim=edge_dim,
            msg_dim=gnn_msg_dim,
            out_dim=gnn_out_dim,
            n_heads=gnn_heads,
            n_layers=gnn_layers,
        )
        self.mlp = MLP(
            hid_sizes=mlp_hid,
            in_dim=gnn_out_dim,
            act=nn.functional.relu,
            act_final=True,
        )
        self.rnn = (
            RNN(
                rnn_cell=rnn_cell,
                input_size=mlp_hid[-1],
                hidden_size=rnn_hidden,
                rnn_layers=rnn_layers,
            )
            if use_rnn
            else None
        )
        
        # end backbone
        
        # This layer squashes the initial weights to prevent gradient vanishing
        self.scale_hid = nn.Linear(rnn_hidden if use_rnn else mlp_hid[-1], scale_hid)
        nn.init.orthogonal_(self.scale_hid.weight) # preserve norm of vectors
        with torch.no_grad():
            self.scale_hid.weight.mul_(scale_final)
        nn.init.zeros_(self.scale_hid.bias)
        
        ## Should be part of TanhNormal class, but nice to have here the networks
        # predict optimal raw action 
        self.mean_head = nn.Linear(scale_hid, action_dim)
        # intended action std
        self.std_head = nn.Linear(scale_hid, action_dim) # will apply softplus to ensure positivity
        for layer in (self.mean_head, self.std_head):
            nn.init.orthogonal_(layer.weight) 
            nn.init.zeros_(layer.bias)
            
    
    def distribution(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents: int,
    ) -> tuple[TanhNormal, Optional[torch.Tensor]]:
        """
           Forward the policy network to get a TanhNormal distribution over actions.
           Returns :
            - TanhNormal distribution
            - rnn_state
        """
        
        x = self.gnn(graph, node_type=0, n_type=n_agents)
        batch_shape = x.shape[:-2]
        x = self.mlp(x).reshape(-1, self.mlp.hid_sizes[-1])
        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)
        else:
            rnn_state = None
        h = self.scale_hid(x)
        
        mean = self.mean_head(h)
        std = F.softplus(self.std_head(h) + self.std_dev_init_inv) + self.std_dev_min
        mean = mean.reshape(batch_shape + (n_agents, -1))
        std = std.reshape(batch_shape + (n_agents, -1))
        return TanhNormal(mean, std), rnn_state
    
    def act(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample an action (or take the mode) and return :
            - action
            - log_prob of the action
            - mode of the generated distribution
            - rnn_state
        """
        dist, rnn_state = self.distribution(graph, rnn_state, n_agents_total)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            action.reshape(-1, action.shape[-1]),
            log_prob.reshape(-1),
            dist.mode().reshape(-1, action.shape[-1]),
            rnn_state,
        )
        
    
    def evaluate(
        self,
        graph: GraphData,
        action: torch.Tensor,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ 
        Evaluates a taken action. Returns :
            - log_prob of the action
            - entropy of the generated distribution
            - rnn_state
        """
        dist, rnn_state = self.distribution(graph, rnn_state, n_agents_total)
        action = action.reshape(dist.mean.shape)
        log_prob = dist.log_prob(action).reshape(-1)
        entropy = dist.entropy().reshape(-1)
        return log_prob, entropy, rnn_state
    
    def enable_training_mode(self, enabled: bool = True) -> None:
        """Called by skrl Agent.enable_models_training_mode()."""
        self.train(enabled)

    @torch.no_grad()
    def initialize_carry(self, n_agents_total: int, device=None) -> Optional[torch.Tensor]:
        if self.rnn is None:
            return None
        return self.rnn.initialize_carry(n_agents_total, device=device)
