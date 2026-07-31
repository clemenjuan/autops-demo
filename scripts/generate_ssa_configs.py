"""Generate the in-scope SSA AO experiment slice.

This is the single source of truth for the committed ``ssa_*_ao_*_n<N>``
configs in ``configs/experiments/``. It emits the approved cheap/mocked AO
backbone from ``docs/ssa_implementation_plan.md`` section 4:

    {ao_symb, ao_rl} x {SAS, CMAS, DMAS, IMAS, HMAS} x N{3,5} = 20 configs

Owner-gated cells are intentionally not emitted here: live LLM ground cells,
PPO/RLlib training runs, world-model cells, and N>5 scale points.

Usage::

    uv run python scripts/generate_ssa_configs.py
    uv run python scripts/generate_ssa_configs.py --output-dir /tmp/ssa-configs
    uv run python scripts/generate_ssa_configs.py --baseline-utility-n1 0.82
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ORGS = {
    "sas": ("sas", {}),
    "cmas": ("centralized_mas", {}),
    "dmas": ("decentralized_mas", {}),
    "imas": ("independent_mas", {"satellite_prefix": "sat"}),
    "hmas": ("hybrid_mas", {"num_clusters": 2}),
}
SIZES = (3, 5)
REPS = ("symb", "rl")
FULL_WEEK_STEPS = 10080
NUM_EPISODES = 30

_PPO_BLOCK = {
    "mechanism": "ppo",
    "mode": "emergent",
    "rollout_fragment": 128,
    "lr": 1.0e-4,
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "clip_ratio": 0.3,
    "ppo_epochs": 30,
    "entropy_coef": 0.01,
    "value_coef": 1.0,
    "max_grad_norm": 0.5,
    "minibatch_size": 256,
    "policy_sharing": {"mode": "shared_all"},
}


def _representation_config(rep: str, org_key: str) -> dict:
    if rep == "symb":
        return {"type": "rule_based_ssa"}
    if rep == "rl":
        # satellite_id is intentionally absent: scoped organisations inject it
        # per agent instance; full-scope organisations control all satellites.
        cfg = {
            "type": "subsymbolic_ssa",
            "rl_mock": True,
            "deterministic": True,
            "checkpoint_path": "",
            "target_count": 100,
            "orbital_period_steps": 94,
            "max_steps": FULL_WEEK_STEPS,
            "compression_time_factor": 2.0,
            "detection_steps": 5,
            "jetson_capacity_mb": 249036.8,
        }
        return cfg
    raise ValueError(f"unknown SSA AO representation cell: {rep}")


def _ppo_block(org_key: str) -> dict:
    block = dict(_PPO_BLOCK)
    block["policy_sharing"] = dict(_PPO_BLOCK["policy_sharing"])
    if org_key == "hmas":
        block["policy_sharing"]["mode"] = "independent_per_agent"
    return block


def _common(
    experiment_id: str,
    org_key: str,
    org_type: str,
    org_config: dict,
    rep: str,
    size: int,
    baseline_utility_n1: float,
) -> dict:
    learned_rl = rep == "rl"
    return {
        "experiment_id": experiment_id,
        "description": (
            f"SSA AO {rep} backbone config for {org_key.upper()} at N={size}."
        ),
        "seed": 42,
        "agent_organization": org_type,
        "decision_procedure": "sda",
        "representation": "rl" if learned_rl else "symbolic",
        "behaviour": "emergent" if learned_rl else "hand_designed",
        "operations_paradigm": "autonomous_onboard",
        "agent_organization_config": dict(org_config),
        "decision_procedure_config": {"sense_encode": True, "decide_act": True},
        "representation_config": _representation_config(rep, org_key),
        "behaviour_config": _ppo_block(org_key) if learned_rl else {"mechanism": "hand_designed", "mode": "hand_designed"},
        "operations_paradigm_config": {"default_mode": "charging"},
        "environment": {
            "constellation_size": size,
            "timestep_seconds": 60,
            "max_steps": FULL_WEEK_STEPS,
            "scenario": "ssa",
            "scenario_config": {
                "scenario_file": "configs/scenarios/ssa.yaml",
                "reward_config": {
                    "local_weight": 0.7,
                    "team_weight": 0.3,
                    "team_reducer": "mean",
                    "collective_weight": 1.0,
                    "mission_scale": 1.0,
                    "collective_negative": True,
                },
            },
        },
        "memory_config": {"history_depth": 100},
        "num_episodes": NUM_EPISODES,
        "max_steps": FULL_WEEK_STEPS,
        "metrics": {
            "enabled": [
                "utility",
                "latency",
                "resource_efficiency",
                "operator_load",
                "explainability",
                "scale_complexity",
                "ssa_coverage",
                "ssa_duplicates",
                "ssa_connectivity",
            ],
            "collection_frequency": "per_step",
            "baseline_utility_n1": float(baseline_utility_n1),
        },
        "output_dir": "data/results/${experiment_id}",
        "save_checkpoints": False,
        "log_level": "INFO",
    }


def build_matrix(baseline_utility_n1: float = 1.0) -> dict[str, dict]:
    """Return {experiment_id: config_dict} for the 20-config SSA AO slice."""
    configs: dict[str, dict] = {}
    for size in SIZES:
        for org_key, (org_type, org_config) in ORGS.items():
            for rep in REPS:
                eid = f"ssa_{org_key}_ao_{rep}_n{size}"
                configs[eid] = _common(
                    experiment_id=eid,
                    org_key=org_key,
                    org_type=org_type,
                    org_config=org_config,
                    rep=rep,
                    size=size,
                    baseline_utility_n1=baseline_utility_n1,
                )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SSA AO experiment configs.")
    parser.add_argument(
        "--output-dir",
        default="configs/experiments/",
        help="Directory to write configs into (default: configs/experiments/).",
    )
    parser.add_argument(
        "--baseline-utility-n1",
        type=float,
        default=1.0,
        help="Placeholder N=1 reference used to compute SSA eta_scale; default 1.0 is not a measured EventSat baseline.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = build_matrix(baseline_utility_n1=args.baseline_utility_n1)
    for eid, cfg in configs.items():
        with (out / f"{eid}.yaml").open("w", encoding="utf-8") as handle:
            yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False, allow_unicode=False)

    print(f"Wrote {len(configs)} SSA AO configs to {out}")


if __name__ == "__main__":
    main()
