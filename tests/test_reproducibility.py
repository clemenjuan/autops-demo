"""
Tests for experiment reproducibility.

Validates that the same configuration and seed produce identical step-level
results, and that different seeds produce different trajectories (via launch
lottery RAAN/ArgP/TA randomization).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config_loader import ExperimentConfig
from src.core.experiment_runner import ExperimentRunner


def _make_config(seed: int, output_dir: str, max_steps: int = 50, num_episodes: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="repro_test",
        seed=seed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        agent_organization="sas",
        decision_procedure="sda",
        representation="symbolic",
        behaviour="hand_designed",
        operations_paradigm="autonomous_hybrid",
        representation_config={"type": "rule_based_eventsat"},
        environment={
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": max_steps,
            "scenario": "eventsat",
            "scenario_config": {},
        },
        save_checkpoints=False,
        log_level="WARNING",
        output_dir=output_dir,
    )


class TestReproducibility:
    def test_same_seed_identical_rewards(self, tmp_path: Path) -> None:
        """Same seed must produce bit-identical rewards and state at every step."""
        cfg1 = _make_config(42, str(tmp_path / "run1"), max_steps=50, num_episodes=2)
        cfg2 = _make_config(42, str(tmp_path / "run2"), max_steps=50, num_episodes=2)

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        assert r1["num_episodes"] == r2["num_episodes"]
        for ep1, ep2 in zip(r1["episodes"], r2["episodes"]):
            assert ep1["num_steps"] == ep2["num_steps"]
            for s1, s2 in zip(ep1["steps"], ep2["steps"]):
                assert s1["rewards"] == s2["rewards"], (
                    f"Reward mismatch at step {s1.get('step')}: "
                    f"{s1['rewards']} vs {s2['rewards']}"
                )
                info1 = s1.get("info", {})
                info2 = s2.get("info", {})
                assert info1.get("battery_soc") == pytest.approx(
                    info2.get("battery_soc"), abs=1e-9
                ), f"Battery SoC mismatch at step {s1.get('step')}"
                assert info1.get("data_downlinked_mb") == pytest.approx(
                    info2.get("data_downlinked_mb"), abs=1e-9
                )
                assert info1.get("resolved_mode") == info2.get("resolved_mode"), (
                    f"Mode mismatch at step {s1.get('step')}: "
                    f"{info1.get('resolved_mode')} vs {info2.get('resolved_mode')}"
                )

    def test_different_seeds_different_trajectories(self, tmp_path: Path) -> None:
        """Different seeds produce different launch-lottery realizations."""
        # Use enough steps that the orbital geometry has time to differ
        cfg1 = _make_config(42, str(tmp_path / "seed42"), max_steps=200)
        cfg2 = _make_config(99, str(tmp_path / "seed99"), max_steps=200)
        for cfg in (cfg1, cfg2):
            cfg.environment.scenario_config = {
                "scenario_file": "configs/scenarios/eventsat.yaml",
                "anomaly_prob": 0.0,
            }

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        episode1 = r1["episodes"][0]
        episode2 = r2["episodes"][0]

        # Assert the exogenous realization directly. Delivery-aligned rewards
        # may legitimately remain equal before either 200-step run delivers
        # data, so reward divergence is not evidence of an orbit lottery.
        lottery_fields = ("raan_deg", "arg_perigee_deg", "true_anomaly_deg")
        elements1 = episode1["orbital_elements"]
        elements2 = episode2["orbital_elements"]
        assert all(field in elements1 and field in elements2 for field in lottery_fields)
        assert any(elements1[field] != elements2[field] for field in lottery_fields), (
            "Seeds 42 and 99 produced identical launch-lottery draws"
        )

    def test_multi_episode_per_seed_deterministic(self, tmp_path: Path) -> None:
        """Each episode in a run uses seed+episode_id deterministically."""
        cfg1 = _make_config(10, str(tmp_path / "run1"), max_steps=30, num_episodes=3)
        cfg2 = _make_config(10, str(tmp_path / "run2"), max_steps=30, num_episodes=3)

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        for ep_idx, (ep1, ep2) in enumerate(zip(r1["episodes"], r2["episodes"])):
            for s1, s2 in zip(ep1["steps"], ep2["steps"]):
                assert s1["rewards"] == s2["rewards"], (
                    f"Episode {ep_idx} reward mismatch at step {s1.get('step')}"
                )

    def test_cross_architecture_anomaly_sync(self, tmp_path: Path) -> None:
        """Anomaly injection must occur at identical steps regardless of ops paradigm.

        Dedicated arrival and duration RNGs are seeded from the episode seed and
        advanced on every environment step, so autonomous (onboard recovery) and
        conventional (ground-gated recovery) cells see the same disturbances.
        """
        SEED = 42
        # Elevated anomaly_prob to guarantee several anomalies in 500 steps
        scenario_config = {"anomaly_prob": 0.05}

        def run(paradigm: str, requires_ground: bool) -> list:
            cfg = _make_config(SEED, str(tmp_path / paradigm), max_steps=500)
            cfg.operations_paradigm = paradigm
            cfg.environment.scenario_config = {
                **scenario_config,
                "anomaly_requires_ground_pass": requires_ground,
            }
            result = ExperimentRunner(config=cfg).run()
            # info.anomaly is the one-step arrival event, including an arrival
            # that refreshes an already-active warning.
            steps_info = [s.get("info", {}) for s in result["episodes"][0]["steps"]]
            return [
                (i, int(info.get("anomaly_duration_steps", 0)))
                for i, info in enumerate(steps_info)
                if bool(info.get("anomaly"))
            ]

        autonomous_injections = run("autonomous_hybrid", requires_ground=False)
        conventional_injections = run("conventional_ground", requires_ground=True)

        assert len(autonomous_injections) > 3, (
            "Too few anomalies injected — increase anomaly_prob or max_steps"
        )
        assert autonomous_injections == conventional_injections, (
            "Paired cells received different anomaly realizations!\n"
            f"  autonomous: {autonomous_injections}\n"
            f"  conventional: {conventional_injections}"
        )
