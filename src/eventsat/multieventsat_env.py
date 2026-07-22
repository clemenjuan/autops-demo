"""
MultiEventsat — a Multi-EventSat constellation scenario.

An integrated constellation environment that owns N per-satellite EventSat dynamics — each
satellite is a full EventSat with its own launch lottery (independent orbital
context), power/data pipeline and reward — and exposes them through the
standard multi-satellite contract:

  * ``ConstellationState`` carrying N satellites (``sat_0 .. sat_{N-1}``), each
    populated with the same ``resources``/``metadata`` keys the RL space adapter
    reads. This lets the EventSat 25D / MultiDiscrete([7,2,2]) adapter and the
    ``autops_actor_critic_v1`` model be reused unchanged.
  * ``StepResult.rewards`` keyed **per satellite** (``{"sat_0": r0, ...}``), so
    the RLlib bridge maps each agent to its own satellite's reward. The
    collective/shared reward term is owned by :class:`MultiEventsatRewardFunction`
    (see ``src/eventsat/rewards.py``) — a future scenario-specific reward
    class is a drop-in replacement, no changes to this env or the bridge.

Design choice (v1): EventSat resources remain per satellite, while the
constellation is stepped and observed as a single environment. Coordination
pressure can enter through the reward team term; subclasses such as SSA can add
shared resources and information flow without changing the bridge contract.

References: Kim et al. (2025) [FVFQ73RF] Independent MAS topology;
Juan Oliver et al. (EUCASS 2025) reward modelling (Individual → Collective).
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import random
from typing import Any, Dict, List

from src.eventsat.rewards import MultiEventsatRewardFunction
from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteEnvironment,
    SatelliteState,
    StepResult,
)
from src.eventsat.env import EventSatEnvironment

logger = logging.getLogger(__name__)


class MultiEventsatEnv(SatelliteEnvironment):
    """N-satellite constellation built from EventSat-class dynamics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.constellation_size = int(config.get("constellation_size", 2))
        self.step_duration_s = float(config.get("step_duration_s", 60.0))
        self.max_steps = int(config.get("max_steps", 10080))
        self.current_step = 0

        self._sat_ids: List[str] = [f"sat_{i}" for i in range(self.constellation_size)]
        # One full EventSat per satellite. Each owns its launch lottery, so the
        # constellation has genuinely distinct eclipse/pass geometry per sat.
        self._subenvs: Dict[str, EventSatEnvironment] = {
            sat_id: EventSatEnvironment(config=dict(config)) for sat_id in self._sat_ids
        }
        self._onboard_compute_active = False

        # Collective reward blend (local + team). Individual per-satellite
        # rewards are produced by the sub-environments themselves.
        proto = next(iter(self._subenvs.values()))
        reward_cfg = {
            **dict(proto.scenario.get("rewards", {}) or {}),
            **dict(config.get("reward_config", {}) or {}),
        }
        self.reward_fn = MultiEventsatRewardFunction(reward_cfg)

        # Constants read by the RL space adapter via getattr(env, name).
        # Identical across sub-envs (same scenario); take them from a prototype.
        self.storage_capacity_mb = proto.storage_capacity_mb
        self.jetson_capacity_mb = proto.jetson_capacity_mb
        self.orbital_period_steps = proto.orbital_period_steps
        self.compression_time_factor = proto.compression_time_factor
        self.detection_steps = proto.detection_steps
        self.battery_capacity_wh = proto.battery_capacity_wh
        # NB: intentionally no ``self.detection_progress`` (it is per-satellite,
        # carried in each satellite's metadata so the adapter reads it per-sat).

    @property
    def onboard_compute_active(self) -> bool:
        return self._onboard_compute_active

    @onboard_compute_active.setter
    def onboard_compute_active(self, active: bool) -> None:
        self._onboard_compute_active = bool(active)
        for sub in self._subenvs.values():
            sub.onboard_compute_active = self._onboard_compute_active

    @property
    def anomaly_requires_ground_pass(self) -> bool:
        return all(sub.anomaly_requires_ground_pass for sub in self._subenvs.values())

    @anomaly_requires_ground_pass.setter
    def anomaly_requires_ground_pass(self, value: bool) -> None:
        # The runner derives this from the operations paradigm's
        # can_self_recover_anomaly(); forward it so every satellite recovers
        # under the same authority rule (sub-envs default to ground-gated).
        for sub in self._subenvs.values():
            sub.anomaly_requires_ground_pass = bool(value)

    def reset(self, seed: int | None = None) -> EnvironmentObservation:
        self.current_step = 0
        plane_overrides = self._constellation_orbit_overrides(seed)
        for idx, sat_id in enumerate(self._sat_ids):
            # Independent launch lottery per satellite: distinct, reproducible
            # seed derived from the episode seed and the satellite index.
            sub_seed = None if seed is None else seed * 1000 + idx
            self._subenvs[sat_id]._orbit_reset_overrides = (
                plane_overrides[idx] if plane_overrides else None
            )
            self._subenvs[sat_id].reset(seed=sub_seed)
        return self.get_observation()

    def _constellation_orbit_overrides(
        self, seed: int | None
    ) -> list[Dict[str, float]] | None:
        """Shared-plane, in-plane-phased elements for all satellites.

        Enabled by a ``constellation_geometry`` config block with
        ``share_plane: true``. One plane (RAAN/ArgP) is drawn per episode from
        the episode seed, and satellites are strung along it with
        ``in_plane_spacing_deg`` true-anomaly offsets — a deployed-constellation
        geometry instead of N independent rideshare lotteries.
        """
        geometry = self.config.get("constellation_geometry") or {}
        if not geometry.get("share_plane", False):
            return None
        rng = random.Random(seed if seed is not None else None)
        raan = rng.uniform(0.0, 360.0)
        argp = rng.uniform(0.0, 360.0)
        ta0 = rng.uniform(0.0, 360.0)
        spacing = float(geometry.get("in_plane_spacing_deg", 5.0))
        return [
            {
                "launch_lottery": False,
                "raan_deg": raan,
                "arg_perigee_deg": argp,
                "true_anomaly_deg": (ta0 + idx * spacing) % 360.0,
            }
            for idx in range(self.constellation_size)
        ]

    def step(self, actions: Dict[str, Any]) -> StepResult:
        if not isinstance(actions, Mapping):
            raise TypeError("MultiEventsat actions must be a satellite-keyed mapping")
        unknown_satellites = sorted(set(actions) - set(self._sat_ids))
        if unknown_satellites:
            raise ValueError(
                "Unknown satellite action keys for MultiEventsat: "
                + ", ".join(unknown_satellites)
            )
        self.current_step += 1
        individual_rewards: Dict[str, float] = {}
        per_satellite_info: Dict[str, Any] = {}
        for sat_id, sub in self._subenvs.items():
            sub_action = {"eventsat_0": actions.get(sat_id, {})}
            sub_result = sub.step(sub_action)
            individual_rewards[sat_id] = float(sub_result.rewards.get("total", 0.0))
            per_satellite_info[sat_id] = sub_result.info

        rewards = self.reward_fn.compute_rewards(individual_rewards)
        info: Dict[str, Any] = {"per_satellite": per_satellite_info}
        # Flat constellation-level aggregates so the EventSat metrics collector
        # (which reads single-satellite info keys) works unchanged here.
        info.update(self._aggregate_info(per_satellite_info))
        return StepResult(
            observation=self.get_observation(),
            rewards=rewards,
            done=self.is_done(),
            truncated=False,
            info=info,
        )

    @staticmethod
    def _aggregate_info(per_satellite_info: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten per-satellite info into constellation-level aggregates.

        Extensive quantities are summed across satellites, intensive ones are
        averaged, and boolean flags are OR-ed. Battery SoC remains an average
        for state telemetry. Per-satellite records remain authoritative for
        rates, AoI, recovery, and command histories; flattening those fields
        loses identity.
        """
        infos = list(per_satellite_info.values())
        if not infos:
            return {}
        n = len(infos)
        sum_keys = (
            "data_stored_mb", "data_downlinked_mb", "observation_hours",
            "total_detections", "undetected_observations",
            "max_achievable_downlink_mb", "total_raw_captured_mb",
            "downlink_raw_equivalent_mb", "jetson_raw_mb",
            "jetson_compressed_mb", "obc_data_mb", "step_downlinked_mb",
            "total_pass_duration_s",
        )
        energy_sum_keys = (
            "gross_energy_consumed_wh",
            "solar_generation_wh",
            "net_battery_depletion_wh",
        )
        mean_keys = ("battery_soc", "prev_battery_soc")
        any_keys = (
            "in_sunlight", "ground_pass_active", "physical_ground_pass_active",
            "contact_window_active", "forced", "in_transition",
            "safety_safe", "constraint_violation",
        )

        agg: Dict[str, Any] = {}
        for key in sum_keys:
            agg[key] = sum(float(i.get(key, 0.0) or 0.0) for i in infos)
        for key in energy_sum_keys:
            # Do not invent explicit zeroes for legacy/manual fixtures: their
            # absence tells the collector to use its SoC-delta compatibility
            # path. New environment transitions always emit all three fields.
            if any(key in i for i in infos):
                agg[key] = sum(float(i.get(key, 0.0) or 0.0) for i in infos)
        for key in mean_keys:
            agg[key] = sum(float(i.get(key, 0.0) or 0.0) for i in infos) / n
        agg["battery_soc_delta_sum"] = sum(
            max(
                0.0,
                float(i.get("prev_battery_soc", i.get("battery_soc", 0.0)) or 0.0)
                - float(i.get("battery_soc", 0.0) or 0.0),
            )
            for i in infos
        )
        for key in any_keys:
            agg[key] = float(any(i.get(key) for i in infos))
        agg["anomaly"] = next(
            (i.get("anomaly") for i in infos if i.get("anomaly")), None
        )
        agg["anomaly_forced_safe"] = float(
            any(i.get("anomaly_forced_safe") for i in infos)
        )
        agg["resolved_mode"] = infos[0].get("resolved_mode")
        agg["requested_modes"] = {
            sat_id: info.get("requested_mode", "")
            for sat_id, info in per_satellite_info.items()
        }
        agg["resolved_modes"] = {
            sat_id: info.get("resolved_mode", "")
            for sat_id, info in per_satellite_info.items()
        }
        return agg

    def get_observation(self) -> EnvironmentObservation:
        satellites: Dict[str, SatelliteState] = {}
        tasks: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        for sat_id, sub in self._subenvs.items():
            sub_obs = sub.get_observation()
            sub_sat = sub_obs.constellation_state.satellites["eventsat_0"]
            metadata = dict(sub_sat.metadata)
            # Per-satellite detection progress (adapter reads it from metadata
            # since this env exposes no env-level detection_progress attribute).
            metadata.setdefault("detection_progress", float(sub.detection_progress))
            satellites[sat_id] = SatelliteState(
                satellite_id=sat_id,
                position=list(sub_sat.position),
                velocity=list(sub_sat.velocity),
                resources=dict(sub_sat.resources),
                status=sub_sat.status,
                metadata=metadata,
            )
            for task in sub_obs.tasks:
                mapped_task = dict(task)
                if mapped_task.get("satellite_id") == "eventsat_0":
                    mapped_task["satellite_id"] = sat_id
                tasks.append(mapped_task)
            events.extend(sub_obs.events)

        constellation = ConstellationState(
            timestep=self.current_step,
            epoch_seconds=self.current_step * self.step_duration_s,
            satellites=satellites,
            global_info={"max_steps": self.max_steps},
        )
        return EnvironmentObservation(
            constellation_state=constellation,
            tasks=tasks,
            events=events,
        )

    def get_metrics(self) -> Dict[str, float]:
        """Aggregate per-satellite metrics across the constellation."""
        totals = {
            "episode_reward": 0.0,
            "total_observation_hours": 0.0,
            "total_downlinked_mb": 0.0,
        }
        for sub in self._subenvs.values():
            sub_metrics = sub.get_metrics()
            for key in totals:
                totals[key] += float(sub_metrics.get(key, 0.0))
        totals["constellation_size"] = float(self.constellation_size)
        return totals
