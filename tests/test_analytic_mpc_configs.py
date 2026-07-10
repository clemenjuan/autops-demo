"""Campaign-contract tests for the intentional analytic CEM-MPC baselines."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.core.config_loader import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments" / "world_model"
HOLDS = (1, 12, 24)


def _path(hold: int, *, jetson: bool) -> Path:
    suffix = "-jetson" if jetson else ""
    return CONFIG_DIR / f"eventsat_sas_ao_cem-analytic-h{hold}{suffix}.yaml"


def _load(hold: int, *, jetson: bool) -> dict:
    return yaml.safe_load(_path(hold, jetson=jetson).read_text(encoding="utf-8"))


def test_exactly_six_analytic_cem_campaign_configs_exist() -> None:
    expected = {
        _path(hold, jetson=jetson).name
        for hold in HOLDS
        for jetson in (False, True)
    }
    actual = {
        path.name
        for path in CONFIG_DIR.glob("eventsat_sas_ao_cem-analytic-*.yaml")
    }
    assert actual == expected


@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize(
    ("jetson", "pricing", "power_w"),
    [(False, "obc", 0.5), (True, "jetson", 7.0)],
)
def test_analytic_cem_campaign_contract(
    hold: int, jetson: bool, pricing: str, power_w: float
) -> None:
    payload = _load(hold, jetson=jetson)
    parsed = ExperimentConfig(**payload)
    representation = payload["representation_config"]

    assert parsed.experiment_id == _path(hold, jetson=jetson).stem
    assert parsed.resolved_onboard_type == "lewm_cem_eventsat"
    assert payload["seed"] == 42
    assert payload["num_episodes"] == 5
    assert payload["max_steps"] == 10080
    assert payload["environment"]["max_steps"] == 10080
    assert payload["environment"]["timestep_seconds"] == 60
    assert payload["environment"]["constellation_size"] == 1

    assert representation["planner_backend"] == "analytic"
    assert representation["planner_pricing"] == pricing
    assert representation["planner_power_w"] == power_w
    assert representation["horizon"] == max(12, hold)
    assert representation["plan_hold"] == hold
    assert representation["samples"] == 256
    assert representation["elites"] == 32
    assert representation["cem_iterations"] == 4
    assert representation["cem_alpha"] == 0.7
    assert representation["reserve_soc"] == 0.5
    assert representation["mission_mode"] == "science"
    assert representation["mission_weights"] == {}
    assert "planner_artifact" not in representation
    assert "strict_artifact" not in representation


@pytest.mark.parametrize("hold", HOLDS)
def test_obc_and_jetson_twins_differ_only_in_pricing(hold: int) -> None:
    obc = deepcopy(_load(hold, jetson=False))
    jetson = deepcopy(_load(hold, jetson=True))

    for payload in (obc, jetson):
        payload.pop("experiment_id")
        payload.pop("description")
        representation = payload["representation_config"]
        representation.pop("planner_pricing")
        representation.pop("planner_power_w")

    assert obc == jetson
