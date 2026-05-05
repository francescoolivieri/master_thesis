from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    level: str
    description: str
    atol: float
    rtol: float
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def default_checkpoint_manifest() -> list[CheckpointSpec]:
    kernel_atol, kernel_rtol = 1e-6, 1e-5
    update_atol, update_rtol = 1e-5, 1e-4

    return [
        CheckpointSpec("kernel/gae/Qhs", "kernel", "Dec-EFOCP GAE safety head output", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/gae/Ql", "kernel", "Dec-EFOCP GAE task scalar output", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/ppo/loss_policy1", "kernel", "Unclipped PPO surrogate term", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/ppo/loss_policy2", "kernel", "Clipped PPO surrogate term", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/ppo/loss_policy", "kernel", "PPO clipped objective", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/ppo/clip_frac", "kernel", "Clip fraction", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/grad/original_norm", "kernel", "Gradient global L2 norm", kernel_atol, kernel_rtol),
        CheckpointSpec("kernel/grad/clipped_norm", "kernel", "Gradient norm after clip", kernel_atol, kernel_rtol),
        CheckpointSpec(
            "actor/rollout/mean", "update", "Policy Normal mean on rollout graphs", update_atol, update_rtol
        ),
        CheckpointSpec("actor/rollout/std", "update", "Policy Normal std on rollout graphs", update_atol, update_rtol),
        CheckpointSpec(
            "actor/rollout/mode", "update", "Squashed policy mode on rollout graphs", update_atol, update_rtol
        ),
        CheckpointSpec(
            "actor/rollout/log_prob", "update", "Policy log-prob of rollout actions", update_atol, update_rtol
        ),
        CheckpointSpec(
            "actor/rollout/fixed_noise",
            "update",
            "Deterministic standard-normal noise used for fixed-noise action checks",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec(
            "actor/rollout/fixed_noise_action",
            "update",
            "Squashed action from rollout mean/std and fixed standard-normal noise",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec(
            "actor/rollout/fixed_noise_log_prob",
            "update",
            "Policy log-prob of the fixed-noise squashed action",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec("update/value/bT_Vl", "update", "Value sequence from Vl scan", update_atol, update_rtol),
        CheckpointSpec("update/value/bTp1_Vl", "update", "Bootstrapped Vl values", update_atol, update_rtol),
        CheckpointSpec("update/value/bTah_Vh", "update", "Safety values from Vh", update_atol, update_rtol),
        CheckpointSpec("update/value/bTp1ah_Vh", "update", "Bootstrapped Vh values", update_atol, update_rtol),
        CheckpointSpec(
            "update/value/bTah_Vh_det",
            "update",
            "Safety values on deterministic rollout",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec(
            "update/value/bTp1ah_Vh_det",
            "update",
            "Bootstrapped Vh values on deterministic rollout",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec("update/gae/bTah_Qh", "update", "GAE safety targets", update_atol, update_rtol),
        CheckpointSpec(
            "update/gae/bTah_Qh_det",
            "update",
            "GAE safety targets for deterministic rollout",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec("update/gae/bT_Ql", "update", "GAE task targets", update_atol, update_rtol),
        CheckpointSpec("update/adv/bT_Al_raw", "update", "Raw task advantage", update_atol, update_rtol),
        CheckpointSpec("update/adv/bT_Al_norm", "update", "Normalized task advantage", update_atol, update_rtol),
        CheckpointSpec("update/adv/bTah_cbf_deriv", "update", "CBF derivative term", update_atol, update_rtol),
        CheckpointSpec("update/adv/bTah_Acbf", "update", "CBF hinge penalty term", update_atol, update_rtol),
        CheckpointSpec("update/adv/bTa_is_safe", "update", "Safety mask", update_atol, update_rtol),
        CheckpointSpec("update/adv/bTa_A", "update", "Final policy advantage", update_atol, update_rtol),
        CheckpointSpec("update/policy/ratio", "update", "Importance-sampling ratio", update_atol, update_rtol),
        CheckpointSpec("update/policy/loss_policy1", "update", "Batch unclipped surrogate", update_atol, update_rtol),
        CheckpointSpec("update/policy/loss_policy2", "update", "Batch clipped surrogate", update_atol, update_rtol),
        CheckpointSpec("update/policy/loss_policy", "update", "Batch surrogate objective", update_atol, update_rtol),
        CheckpointSpec("update/policy/clip_frac", "update", "Batch clip fraction", update_atol, update_rtol),
        CheckpointSpec("update/policy/entropy", "update", "Batch policy entropy", update_atol, update_rtol),
        CheckpointSpec(
            "update/policy/policy_loss", "update", "Batch policy loss with entropy term", update_atol, update_rtol
        ),
        CheckpointSpec("update/loss/Vl_global", "update", "Global pre-update Vl L2 loss", update_atol, update_rtol),
        CheckpointSpec(
            "update/loss/Vh_det_global",
            "update",
            "Global pre-update deterministic Vh L2 loss",
            update_atol,
            update_rtol,
        ),
        CheckpointSpec("update/metrics/policy_loss", "update", "Logged policy loss", update_atol, update_rtol),
        CheckpointSpec("update/metrics/Vl_loss", "update", "Logged Vl loss", update_atol, update_rtol),
        CheckpointSpec("update/metrics/Vh_loss", "update", "Logged Vh loss", update_atol, update_rtol),
        CheckpointSpec(
            "update/metrics/policy_grad_norm", "update", "Logged policy grad norm", update_atol, update_rtol
        ),
        CheckpointSpec("update/metrics/Vl_grad_norm", "update", "Logged Vl grad norm", update_atol, update_rtol),
        CheckpointSpec("update/metrics/Vh_grad_Vh_norm", "update", "Logged Vh grad norm", update_atol, update_rtol),
        CheckpointSpec(
            "update/param_delta/policy", "update", "L2 norm of policy parameter delta", update_atol, update_rtol
        ),
        CheckpointSpec("update/param_delta/Vl", "update", "L2 norm of Vl parameter delta", update_atol, update_rtol),
        CheckpointSpec("update/param_delta/Vh", "update", "L2 norm of Vh parameter delta", update_atol, update_rtol),
    ]
