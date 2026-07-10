"""Semantic tests for the dedicated world-model planner board."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_wm_board as board


def _write_run(
    root: Path,
    run_id: str,
    *,
    representation_config: dict,
    mean: dict,
    steps: int = board.FULL_WEEK_STEPS,
    episodes: int = 5,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    config = {
        "experiment_id": run_id,
        "max_steps": steps,
        "representation_config": representation_config,
    }
    results = {
        "experiment_id": run_id,
        "episodes": [{"episode": i} for i in range(episodes)],
        "experiment_statistics": {"mean": mean},
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "results.json").write_text(json.dumps(results))


def test_collect_keeps_latent_analytic_fallback_and_reference_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "type": "lewm_cem_eventsat",
        "horizon": 12,
        "plan_hold": 12,
        "samples": 256,
        "cem_iterations": 4,
        "mission_mode": "science",
    }
    _write_run(
        tmp_path,
        "latent",
        representation_config={**common, "planner_backend": "latent"},
        mean={
            "rollout_backend": "latent",
            "artifact_loaded": 1.0,
            "artifact_fallback": 0.0,
            "utility": 1.0,
        },
    )
    _write_run(
        tmp_path,
        "analytic",
        representation_config={
            **common,
            "planner_backend": "analytic",
            "planner_pricing": "obc",
            "planner_power_w": 0.5,
        },
        mean={
            "rollout_backend": "analytic",
            "artifact_loaded": 0.0,
            "artifact_fallback": 0.0,
            "utility": 1.1,
        },
    )
    # Runtime truth must override the intended backend. This row is not analytic.
    _write_run(
        tmp_path,
        "failed-artifact",
        representation_config={**common, "planner_backend": "analytic"},
        mean={
            "rollout_backend": "fallback",
            "artifact_loaded": 0.0,
            "artifact_fallback": 1.0,
            "utility": 99.0,
        },
    )
    _write_run(
        tmp_path,
        "eventsat_sas_ao_symb",
        representation_config={"type": "rule_based_eventsat"},
        mean={"utility": 1.2, "utility_fraction_of_physical_ceiling": 0.9},
    )
    monkeypatch.setattr(board, "RESULTS", tmp_path)

    wm, refs = board.collect()

    assert [row["id"] for row in wm] == ["latent", "analytic", "failed-artifact"]
    assert [board._rollout_backend(row) for row in wm] == [
        "latent",
        "analytic",
        "fallback",
    ]
    assert [row["status"] for row in wm] == ["measured", "measured", "fallback"]
    assert len(refs) == 1
    assert board._rollout_backend(refs[0], is_ref=True) == "rule-based reference"
    assert refs[0]["status"] == "measured"

    # The numerically dominant fallback must remain excluded from best-run marking.
    assert board._best(wm)["utility"] == "analytic"


def test_render_has_backend_cost_and_ceiling_columns() -> None:
    analytic = {
        "id": "analytic-obc",
        "rc": {
            "type": "lewm_cem_eventsat",
            "planner_backend": "analytic",
            "planner_pricing": "obc",
            "planner_power_w": 0.5,
            "horizon": 12,
            "plan_hold": 12,
            "samples": 256,
            "cem_iterations": 4,
            "mission_mode": "science",
        },
        "compute_w": 0.5,
        "reported_backend": "analytic",
        "mean": {
            "rollout_backend": "analytic",
            "planner_ms_per_event": 12.345,
            "planner_energy_wh": 0.125,
            "utility": 0.75,
            "utility_fraction_of_physical_ceiling": 0.5,
            "artifact_loaded": 0.0,
            "artifact_fallback": 0.0,
        },
        "n": 5,
        "steps": board.FULL_WEEK_STEPS,
        "socs": [],
        "status": "measured",
    }
    fallback = {
        **analytic,
        "id": "failed-latent",
        "reported_backend": "fallback",
        "mean": {
            **analytic["mean"],
            "rollout_backend": "fallback",
            "artifact_fallback": 1.0,
            "utility": 10.0,
        },
        "status": "fallback",
    }
    reference = {
        "id": "eventsat_sas_ao_symb",
        "ref_label": board.REFERENCE_IDS["eventsat_sas_ao_symb"],
        "rc": {"type": "rule_based_eventsat"},
        "compute_w": 0.0,
        "mean": {"utility": 0.7, "utility_fraction_of_physical_ceiling": 0.48},
        "n": 5,
        "steps": board.FULL_WEEK_STEPS,
        "socs": [],
        "status": "measured",
    }

    html = board.render([analytic, fallback], [reference])

    for heading in (
        "Rollout backend",
        "Planner ms / event",
        "Planner Wh / episode",
        "Utility",
        "Utility / physical ceiling",
    ):
        assert f"<th>{heading}</th>" in html
    assert 'data-backend="analytic"' in html
    assert 'data-backend="fallback"' in html
    assert 'data-backend="rule-based-reference"' in html
    assert "12.35" in html
    assert "0.125" in html
    assert "0.500" in html
    assert "fallback — excluded" in html
    assert "intentional analytic" in html
    assert "sub-watt OBC" in html
    assert "never grouped with analytic" in html


def test_board_prefers_canonical_latency_but_reads_legacy_seconds() -> None:
    canonical = {
        "mean": {"planner_ms_per_event": 3.5, "planner_latency_s": 9.0}
    }
    legacy = {"mean": {"planner_latency_s": 0.012}}

    assert board._metric_value(canonical, "planner_ms_per_event") == pytest.approx(3.5)
    assert board._metric_value(legacy, "planner_ms_per_event") == pytest.approx(12.0)


def test_explicit_fallback_flag_overrides_contradictory_backend_label() -> None:
    row = {
        "rc": {"type": "lewm_cem_eventsat", "planner_backend": "analytic"},
        "reported_backend": "analytic",
        "mean": {"artifact_loaded": 0.0, "artifact_fallback": 1.0},
    }

    assert board._rollout_backend(row) == "fallback"
