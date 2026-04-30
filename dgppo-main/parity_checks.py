#!/usr/bin/env python3
import argparse
from pathlib import Path

from dgppo.parity.determinism import DeterminismContract
from dgppo.parity.io import save_json
from dgppo.parity.kernel_fixtures import KernelFixtureConfig, export_kernel_fixtures
from dgppo.parity.manifest import default_checkpoint_manifest
from dgppo.parity.reference_traces import UpdateFixtureConfig, export_drift_trace, export_update_fixtures
from dgppo.parity.report import compare_npz, format_report_text


def _build_contract(args: argparse.Namespace) -> DeterminismContract:
    return DeterminismContract(
        seed=args.seed,
        dtype="float32",
        force_cpu=args.force_cpu,
        jax_disable_jit=args.jax_disable_jit,
        reset_numpy_seed_before_update=not args.no_reset_numpy_seed_before_update,
    )


def _build_update_config(args: argparse.Namespace) -> UpdateFixtureConfig:
    n_drift_updates = getattr(args, "n_drift_updates", 4)
    return UpdateFixtureConfig(
        seed=args.seed,
        env_id=args.env,
        num_agents=args.num_agents,
        obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        n_env_train=args.n_env_train,
        batch_size=args.batch_size,
        rnn_step=args.rnn_step,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        lr_actor=args.lr_actor,
        lr_Vl=args.lr_Vl,
        lr_Vh=args.lr_Vh,
        clip_eps=args.clip_eps,
        coef_ent=args.coef_ent,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        cbf_weight=args.cbf_weight,
        cbf_schedule=not args.no_cbf_schedule,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        cost_weight=args.cost_weight,
        cost_schedule=args.cost_schedule,
        train_steps=args.train_steps,
        epoch_ppo=args.epoch_ppo,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        update_step=args.update_step,
        n_drift_updates=n_drift_updates,
    )


def _add_shared_update_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="LidarSpread")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=2)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--n-env-train", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-step", type=int, default=16)
    parser.add_argument("--actor-gnn-layers", type=int, default=2)
    parser.add_argument("--Vl-gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-Vl", type=float, default=1e-3)
    parser.add_argument("--lr-Vh", type=float, default=1e-3)
    parser.add_argument("--clip-eps", type=float, default=0.25)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--no-cbf-schedule", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--cost-schedule", action="store_true", default=False)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--epoch-ppo", type=int, default=1)
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument("--jax-disable-jit", action="store_true", default=False)
    parser.add_argument("--no-reset-numpy-seed-before-update", action="store_true", default=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="DG-PPO parity fixture/export/comparison tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("export-manifest", help="Export checkpoint and tolerance manifest.")
    p_manifest.add_argument("--output", type=Path, default=Path("parity_artifacts/manifest.json"))

    p_kernel = sub.add_parser("export-kernel", help="Export kernel-level parity fixtures.")
    p_kernel.add_argument("--output-dir", type=Path, default=Path("parity_artifacts/kernel"))
    p_kernel.add_argument("--seed", type=int, default=0)
    p_kernel.add_argument("--T", type=int, default=5)
    p_kernel.add_argument("--num-agents", type=int, default=4)
    p_kernel.add_argument("--n-cost", type=int, default=2)
    p_kernel.add_argument("--disc-gamma", type=float, default=0.99)
    p_kernel.add_argument("--gae-lambda", type=float, default=0.95)
    p_kernel.add_argument("--clip-eps", type=float, default=0.25)
    p_kernel.add_argument("--max-grad-norm", type=float, default=2.0)

    p_update = sub.add_parser("export-update", help="Export single-update DG-PPO fixture and checkpoints.")
    p_update.add_argument("--output-dir", type=Path, default=Path("parity_artifacts/update"))
    p_update.add_argument("--update-step", type=int, default=0)
    _add_shared_update_args(p_update)

    p_drift = sub.add_parser("export-drift", help="Export short deterministic multi-update drift trace.")
    p_drift.add_argument("--output-dir", type=Path, default=Path("parity_artifacts/drift"))
    p_drift.add_argument("--n-drift-updates", type=int, default=4)
    p_drift.add_argument("--update-step", type=int, default=0)
    _add_shared_update_args(p_drift)

    p_compare = sub.add_parser("compare", help="Compare expected and actual npz outputs.")
    p_compare.add_argument("--expected-npz", type=Path, required=True)
    p_compare.add_argument("--actual-npz", type=Path, required=True)
    p_compare.add_argument("--output-json", type=Path, default=Path("parity_artifacts/comparison_report.json"))
    p_compare.add_argument("--output-text", type=Path, default=Path("parity_artifacts/comparison_report.txt"))

    args = parser.parse_args()

    if args.command == "export-manifest":
        manifest = [spec.to_dict() for spec in default_checkpoint_manifest()]
        save_json(args.output, {"manifest": manifest})
        print(f"Wrote manifest to {args.output}")
        return

    if args.command == "export-kernel":
        cfg = KernelFixtureConfig(
            seed=args.seed,
            T=args.T,
            n_agents=args.num_agents,
            n_cost=args.n_cost,
            disc_gamma=args.disc_gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            max_grad_norm=args.max_grad_norm,
        )
        out = export_kernel_fixtures(args.output_dir, cfg)
        print(f"Wrote kernel fixtures: {out}")
        return

    if args.command == "export-update":
        cfg = _build_update_config(args)
        contract = _build_contract(args)
        out = export_update_fixtures(args.output_dir, cfg, contract)
        print(f"Wrote update fixture: {out}")
        return

    if args.command == "export-drift":
        cfg = _build_update_config(args)
        contract = _build_contract(args)
        out = export_drift_trace(args.output_dir, cfg, contract)
        print(f"Wrote drift trace: {out}")
        return

    if args.command == "compare":
        manifest = default_checkpoint_manifest()
        report = compare_npz(
            expected_npz=args.expected_npz,
            actual_npz=args.actual_npz,
            manifest=manifest,
            output_json=args.output_json,
        )
        text = format_report_text(report)
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(text, encoding="utf-8")
        print(text)
        print(f"Report JSON: {args.output_json}")
        print(f"Report TXT:  {args.output_text}")
        return


if __name__ == "__main__":
    main()
