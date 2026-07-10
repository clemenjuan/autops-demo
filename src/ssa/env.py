"""SSA constellation environment.

Kinematics: each satellite carries real orbital elements (taken from its
EventSat sub-environment's per-episode orbit, including shared-plane
constellation phasing when ``constellation_geometry.share_plane`` is set) and
is propagated for detection/ISL geometry. Two-body propagation is the default
position backend — differential J2 between near-identical SSO orbits is
negligible over a week, and the per-step cost of Orekit JVM calls for
N sats + M targets is not. Orekit still owns eclipse/pass windows via the
sub-environments' orbital contexts.

ISL: ``isl_share`` transfers custody of undelivered observation records to a
feasible neighbour, rate-limited by the UHF/QPSK link budget integrated over
sub-minute feasibility windows (mirroring how ground passes are resolved at
second accuracy inside a 60 s step). Knowledge (detection matrix + best
estimates) still merges freely — it is N×M bits, physically negligible next to
record payloads.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
    StepResult,
)
from src.eventsat.multieventsat_env import MultiEventsatEnv
from src.orbital.isl import (
    ISLConfig,
    effective_data_rate_bps,
    vector_range_km,
)
from src.ssa.metrics import geometric_utility_ceiling
from src.ssa.rewards import SSARewardFunction
from src.ssa.targets import (
    _EARTH_RADIUS_KM as EARTH_RADIUS_KM,
    DetectionAccess,
    RSOTarget,
    detection_draw,
    generate_sso_catalog,
    optical_accesses,
    propagate_rso_position_km,
    propagated_catalog_positions_km,
    sun_unit_eci,
)


SSA_MODES = [
    "charging",
    "communication",
    "payload_observe",
    "payload_detect",
    "isl_share",
    "safe",
]

# Matches the default orbital-context epoch (src/orbital/context.py) so that
# detection geometry and eclipse/pass windows describe the same sky.
_ELEMENT_EPOCH = datetime(2026, 6, 1, tzinfo=timezone.utc)


class SSAEnvironment(MultiEventsatEnv):
    """Multi-EventSat plus RSO detection, ISL record relay, and ground archive."""

    def __init__(self, config: Dict[str, Any]) -> None:
        scenario_defaults = self._load_ssa_scenario_defaults(config)
        # The shared EventSat dynamics currently name their radio compatibility
        # key "sband". Keep the canonical SSA scenario truthfully named
        # "xband" while presenting the same rate block to that generic base.
        xband = (
            scenario_defaults.get("communications", {}).get("xband", {})
            if scenario_defaults
            else {}
        )
        if xband:
            scenario_overrides = deepcopy(config.get("scenario_overrides", {}))
            communications = dict(scenario_overrides.get("communications", {}))
            communications.setdefault("sband", dict(xband))
            scenario_overrides["communications"] = communications
            config = {**config, "scenario_overrides": scenario_overrides}
        self.ssa_config = {**dict(scenario_defaults.get("ssa", {})), **dict(config.get("ssa", {}))}
        self.targets_config = {
            **dict(scenario_defaults.get("targets", {})),
            **dict(config.get("targets", {})),
        }
        self.isl_config_block = {
            **dict(scenario_defaults.get("isl", {})),
            **dict(config.get("isl", {})),
        }
        self.ground_config = {
            **dict(scenario_defaults.get("ground_station", {})),
            **dict(config.get("ground_station", {})),
        }
        satellite_positions = {
            **dict(scenario_defaults.get("satellite_positions_km", {})),
            **dict(config.get("satellite_positions_km", {})),
        }
        # Static positions are a test/toy fixture only; when present they pin
        # the geometry and disable orbital kinematics for those satellites.
        self._configured_sat_positions = {
            str(k): tuple(float(x) for x in v)
            for k, v in satellite_positions.items()
        }
        if "constellation_geometry" not in config:
            geometry = scenario_defaults.get("constellation_geometry")
            if geometry:
                config = {**config, "constellation_geometry": dict(geometry)}
        super().__init__(config)

        # ``MultiEventsatEnv.step`` applies ``self.reward_fn`` once to the
        # individual satellite rewards.  Keep that base aggregator in place;
        # SSA's delivered-coverage term is added separately only after physical
        # record delivery for this step is known.  Pre-fix SSA rewards applied
        # the local/team blend twice and are therefore not comparable.
        self.ssa_reward_fn = SSARewardFunction(config.get("reward_config", {}))
        self.mode_list = list(SSA_MODES)
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.mode_list)}
        self.target_count = int(self.targets_config.get("count", 6))
        self.fov_half_angle_deg = float(
            self.targets_config.get("fov_half_angle_deg", 1.9)
        )
        self.boresight_pitch_deg = float(
            self.targets_config.get("boresight_pitch_deg", 12.0)
        )
        self.r_cap_km = float(self.targets_config.get("r_cap_km", 150.0))
        self.m_lim = float(self.targets_config.get("m_lim", 15.0))
        self.sigma_m = float(self.targets_config.get("sigma_m", 0.5))
        self.albedo = float(self.targets_config.get("albedo", 0.13))
        self.catalog_seed = self.targets_config.get("seed")
        self.prefer_orekit_targets = bool(self.targets_config.get("prefer_orekit", False))
        self.raan_spread_deg = float(self.targets_config.get("raan_spread_deg", 0.6))
        self.fixed_target_positions = self._parse_fixed_target_positions(
            self.targets_config.get("fixed_positions_km")
        )
        if self.fixed_target_positions:
            self.target_count = len(self.fixed_target_positions)
        self.target_ids = [f"rso_{idx}" for idx in range(self.target_count)]
        self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}
        self.targets: list[RSOTarget] = []
        self._episode_seed = 0

        self.isl_config = ISLConfig(**{
            key: value for key, value in self.isl_config_block.items()
            if key in ISLConfig.__dataclass_fields__
        })
        self.isl_relay = bool(self.ssa_config.get("isl_relay", True))
        self.isl_unicast = bool(self.isl_config_block.get("unicast", True))
        self.isl_substep_s = float(self.isl_config_block.get("substep_resolution_s", 10.0))
        self.isl_power_overhead_w = float(self.isl_config_block.get("power_overhead_w", 5.0))
        self.isl_min_soc = float(self.ssa_config.get("isl_min_soc", 0.3))
        self.record_size_bytes = float(self.ssa_config.get("record_size_kb", 1.0)) * 1024.0
        self.prefer_orekit_positions = bool(self.ssa_config.get("prefer_orekit_positions", False))
        self.ground_always_visible = bool(self.ground_config.get("always_visible", False))

        self._sat_orbits: dict[str, RSOTarget] = {}
        self.detection_matrix: list[list[int]] = []
        self.onboard_estimates: dict[str, dict[str, dict[str, Any]]] = {}
        self.ground_archive: dict[str, list[dict[str, Any]]] = {}
        # Undelivered buffer holds at most one record per object per satellite
        # (the best-quality track) — bounded memory, "forward the current best
        # estimate" semantics.
        self._undelivered_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._last_observed_step: dict[str, int] = {}
        self._last_detection_by_sat: dict[tuple[str, str], int] = {}
        self._first_detected_step: dict[str, int] = {}
        self._delivered_step: dict[str, int] = {}
        self._revisit_intervals: list[int] = []
        self._relayed_first_deliveries = 0
        self._coverage_auc_sum = 0.0
        self.duplicate_observations = 0
        self.successful_observations = 0
        self.total_observation_records = 0
        self.isl_attempts = 0
        self.isl_successes = 0
        self.isl_records_relayed = 0
        self.isl_bytes_transferred = 0.0
        self.last_step_detections: dict[str, list[str]] = {}
        self.last_step_downlinked_records = 0
        self.physical_utility_ceiling = 0.0

    @staticmethod
    def _load_ssa_scenario_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
        scenario_path = config.get("scenario_config") or config.get("scenario_file")
        if not scenario_path:
            return {}
        path = Path(str(scenario_path))
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle) or {}
        return scenario if isinstance(scenario, dict) else {}

    def reset(self, seed: int | None = None) -> EnvironmentObservation:
        self._episode_seed = int(seed) if seed is not None else 0
        super().reset(seed=seed)
        self._sat_orbits = {
            sat_id: self._sat_orbit_elements(sat_id)
            for sat_id in self._sat_ids
        }
        catalog_seed = self.catalog_seed if self.catalog_seed is not None else seed
        geometry = self.config.get("constellation_geometry") or {}
        raan_center = None
        if geometry.get("share_plane") and self._sat_orbits:
            raan_center = self._sat_orbits[self._sat_ids[0]].raan_deg
        self.targets = generate_sso_catalog(
            self.target_count,
            seed=None if catalog_seed is None else int(catalog_seed),
            altitude_range_km=tuple(self.targets_config.get("altitude_range_km", (600.0, 900.0))),
            eccentricity_max=float(self.targets_config.get("eccentricity_max", 0.001)),
            inclination_range_deg=tuple(self.targets_config.get("inclination_range_deg", (97.0, 99.0))),
            object_size_m=float(self.targets_config.get("object_size_m", 1.0)),
            epoch=_ELEMENT_EPOCH,
            raan_center_deg=raan_center,
            raan_spread_deg=self.raan_spread_deg,
        )
        if self.fixed_target_positions:
            self.target_ids = list(self.fixed_target_positions.keys())
            self.target_index = {
                target_id: idx for idx, target_id in enumerate(self.target_ids)
            }
            self.targets = [
                replace(target, object_id=target_id)
                for target, target_id in zip(self.targets, self.target_ids)
            ]
        else:
            self.target_ids = [target.object_id for target in self.targets]
            self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}

        self.detection_matrix = [
            [0 for _ in range(self.target_count)] for _ in range(self.constellation_size)
        ]
        self.onboard_estimates = {sat_id: {} for sat_id in self._sat_ids}
        self.ground_archive = {target_id: [] for target_id in self.target_ids}
        self._undelivered_records = {sat_id: {} for sat_id in self._sat_ids}
        self._last_observed_step = {}
        self._last_detection_by_sat = {}
        self._first_detected_step = {}
        self._delivered_step = {}
        self._revisit_intervals = []
        self._relayed_first_deliveries = 0
        self._coverage_auc_sum = 0.0
        self.duplicate_observations = 0
        self.successful_observations = 0
        self.total_observation_records = 0
        self.isl_attempts = 0
        self.isl_successes = 0
        self.isl_records_relayed = 0
        self.isl_bytes_transferred = 0.0
        self.last_step_detections = {sat_id: [] for sat_id in self._sat_ids}
        self.last_step_downlinked_records = 0
        self.physical_utility_ceiling = self._compute_physical_utility_ceiling()
        return self.get_observation()

    def _sat_orbit_elements(self, sat_id: str) -> RSOTarget:
        orbit = getattr(self._subenvs[sat_id], "_episode_orbit", None) or {}
        return RSOTarget(
            object_id=sat_id,
            semi_major_axis_km=EARTH_RADIUS_KM + float(orbit.get("altitude_km", 775.0)),
            eccentricity=float(orbit.get("eccentricity", 0.001)),
            inclination_deg=float(orbit.get("inclination_deg", 98.6)),
            raan_deg=float(orbit.get("raan_deg", 0.0)),
            arg_perigee_deg=float(orbit.get("arg_perigee_deg", 0.0)),
            true_anomaly_deg=float(orbit.get("true_anomaly_deg", 0.0)),
            epoch=_ELEMENT_EPOCH,
        )

    def _compute_physical_utility_ceiling(self) -> float:
        """Agent-free geometric ceiling for deliverable SSA coverage."""
        return geometric_utility_ceiling(
            self._ground_pass_windows_for_ceiling(),
            self._visibility_timeline(),
            self.target_count,
        )

    def _ground_pass_windows_for_ceiling(self) -> list[dict[str, Any]]:
        if self.ground_always_visible:
            final_step = max(0, self.max_steps - 1)
            return [{"start_step": final_step, "end_step": final_step}]

        windows: list[dict[str, Any]] = []
        for sat_id, sub in self._subenvs.items():
            if not hasattr(sub, "get_ground_passes"):
                continue
            for window in sub.get_ground_passes():
                try:
                    start = int(window.get("start_step"))
                    end = int(window.get("end_step", start))
                except (AttributeError, TypeError, ValueError):
                    continue
                if end < 0 or start >= self.max_steps:
                    continue
                # ``end_step`` is a backwards-compatible coarse field.  The
                # simplified pass generator historically stores the first step
                # after a short pass there, so derive the final step with actual
                # contact from the second-accurate window when available.
                try:
                    start_s = float(window.get("start_s"))
                    end_s = float(window.get("end_s"))
                except (AttributeError, TypeError, ValueError):
                    start_s = end_s = 0.0
                if end_s > start_s and self.step_duration_s > 0.0:
                    end = max(start, math.ceil(end_s / self.step_duration_s) - 1)
                normalized = dict(window)
                normalized["start_step"] = max(0, start)
                normalized["end_step"] = min(self.max_steps - 1, end)
                normalized["satellite_id"] = sat_id
                windows.append(normalized)
        return windows

    def _visibility_timeline(self):
        for step in range(self.max_steps):
            epoch_seconds = step * self.step_duration_s
            target_positions = self._target_positions_at(epoch_seconds)
            visible: set[str] = set()
            for sat_id in self._sat_ids:
                accesses = self._optical_accesses_at(
                    sat_id,
                    epoch_seconds,
                    target_positions=target_positions,
                )
                visible.update(access.object_id for access in accesses)
            yield {"step": step, "visible_target_ids": sorted(visible)}

    def step(self, actions: Dict[str, Any]) -> StepResult:
        # MultiEventsatEnv increments its constellation counter before returning.
        # Preserve the action's pre-increment timestep so SSA detection geometry
        # matches the sub-environments and the 0..N-1 ceiling timeline.
        action_step = self.current_step
        action_epoch_s = action_step * self.step_duration_s
        decoded = self._decode_actions(actions)
        event_actions = {
            sat_id: {"mode": ("charging" if mode == "isl_share" else mode)}
            for sat_id, mode in decoded.items()
        }
        base_result = super().step(event_actions)
        per_sat_info = dict(base_result.info.get("per_satellite", {}))

        self.last_step_detections = {sat_id: [] for sat_id in self._sat_ids}
        self.last_step_downlinked_records = 0
        target_positions = self._target_positions_at(action_epoch_s)

        for sat_id, requested_mode in decoded.items():
            info = per_sat_info.get(sat_id, {})
            productive_observe = (
                requested_mode == "payload_observe"
                and info.get("resolved_mode") == "payload_observe"
                and not bool(info.get("in_transition", False))
                and bool(info.get("observation_accepted", False))
            )
            if productive_observe:
                accesses = self._optical_accesses_at(
                    sat_id,
                    action_epoch_s,
                    target_positions=target_positions,
                )
                for access in accesses:
                    draw = detection_draw(
                        self._episode_seed,
                        access.object_id,
                        sat_id,
                        action_step,
                    )
                    if draw < access.p_detect:
                        self._record_detection(sat_id, access, step=action_step)

        isl_energy_by_sat = self._apply_isl_shares(decoded)
        if isl_energy_by_sat:
            # The base constellation snapshot predates SSA's extra radio draw.
            # Refresh state telemetry and add the radio to gross load. Archived
            # net-SoC energy results are not comparable with this explicit
            # gross-load contract.
            refreshed_per_sat: dict[str, dict[str, Any]] = {}
            for sat_id, raw_info in per_sat_info.items():
                sat_info = dict(raw_info)
                sat_info["battery_soc"] = float(self._subenvs[sat_id].battery_soc)
                isl_energy_wh = float(isl_energy_by_sat.get(sat_id, 0.0))
                sat_info["isl_energy_consumed_wh"] = isl_energy_wh
                sat_info["gross_energy_consumed_wh"] = float(
                    sat_info.get("gross_energy_consumed_wh", 0.0) or 0.0
                ) + isl_energy_wh
                prev_soc = float(
                    sat_info.get("prev_battery_soc", sat_info["battery_soc"])
                    or sat_info["battery_soc"]
                )
                sat_info["net_battery_depletion_wh"] = max(
                    0.0,
                    (prev_soc - sat_info["battery_soc"])
                    * self._subenvs[sat_id].battery_capacity_wh,
                )
                refreshed_per_sat[sat_id] = sat_info
            per_sat_info = refreshed_per_sat

            total_isl_energy_wh = sum(isl_energy_by_sat.values())
            aggregate = self._aggregate_info(per_sat_info)
            aggregate["isl_energy_consumed_wh"] = total_isl_energy_wh
        self._apply_ground_downlinks(decoded, per_sat_info)
        self._coverage_auc_sum += self.delivered_coverage

        info = dict(base_result.info)
        if isl_energy_by_sat:
            info["per_satellite"] = per_sat_info
            info.update(aggregate)
        info.update(self._ssa_info(decoded))
        rewards = self.ssa_reward_fn.add_mission_term(
            base_result.rewards,
            delivered_coverage=self.delivered_coverage,
        )
        return StepResult(
            observation=self.get_observation(),
            rewards=rewards,
            done=base_result.done,
            truncated=base_result.truncated,
            info=info,
        )

    def get_observation(self) -> EnvironmentObservation:
        obs = super().get_observation()
        target_positions = self._target_positions() if self.target_ids else {}
        satellites: dict[str, SatelliteState] = {}
        tasks = [
            dict(task)
            for task in obs.tasks
            if task.get("type") != "observe_rso"
        ]
        epoch_seconds = self.current_step * self.step_duration_s
        for sat_id, sat in obs.constellation_state.satellites.items():
            position = self._satellite_position(sat_id)
            known_objects = set(self.onboard_estimates.get(sat_id, {}))
            cued_accesses = self._optical_accesses_at(
                sat_id,
                epoch_seconds,
                target_positions=target_positions,
                restrict_ids=known_objects,
            )
            known_ages: dict[str, int] = {}
            for object_id in sorted(known_objects):
                record = self.onboard_estimates[sat_id][object_id]
                refreshed = int(
                    record.get(
                        "last_refresh_step",
                        record.get("time_step", self.current_step),
                    )
                )
                known_ages[object_id] = max(0, self.current_step - refreshed)

            metadata = dict(sat.metadata)
            sat_idx = self._sat_index(sat_id)
            metadata.update({
                "ssa_detection_matrix": deepcopy(self.detection_matrix),
                "ssa_detection_row": (
                    list(self.detection_matrix[sat_idx])
                    if self.detection_matrix
                    else []
                ),
                "ssa_known_objects": sorted(known_objects),
                "ssa_delivered_objects": sorted(self.delivered_object_ids),
                "ssa_undelivered_records": len(
                    self._undelivered_records.get(sat_id, {})
                ),
                "ssa_predicted_in_fov": [
                    access.object_id for access in cued_accesses
                ],
                "ssa_known_object_ages": known_ages,
                "ssa_onboard_coverage": self.onboard_coverage,
                "ssa_delivered_coverage": self.delivered_coverage,
            })
            if self.ground_always_visible:
                # The toy/fixture ground model is continuous contact, not an
                # empty pass schedule. Expose that configured truth and its
                # remaining physical capacity so controller admission logic
                # does not interpret it as a zero-capacity mission.
                remaining_steps = max(0, self.max_steps - self.current_step)
                rate_kbps = float(self._subenvs[sat_id].downlink_rate_kbps)
                remaining_capacity_mb = (
                    rate_kbps
                    * remaining_steps
                    * self.step_duration_s
                    / 8000.0
                )
                metadata.update({
                    "ground_pass_active": True,
                    "contact_window_active": True,
                    "physical_ground_pass_active": True,
                    "achievable_downlink_mb": remaining_capacity_mb,
                    "next_future_pass_downlink_mb": remaining_capacity_mb,
                    "second_future_pass_downlink_mb": remaining_capacity_mb,
                })
            satellites[sat_id] = SatelliteState(
                satellite_id=sat_id,
                position=list(position),
                velocity=list(sat.velocity),
                resources=dict(sat.resources),
                status=sat.status,
                metadata=metadata,
            )

        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=obs.constellation_state.timestep,
                epoch_seconds=obs.constellation_state.epoch_seconds,
                satellites=satellites,
                global_info={
                    **dict(obs.constellation_state.global_info),
                    "ssa_detection_matrix": deepcopy(self.detection_matrix),
                    "ssa_target_count": self.target_count,
                    "ssa_onboard_coverage": self.onboard_coverage,
                    "ssa_delivered_coverage": self.delivered_coverage,
                    "ssa_ground_archive_records": sum(
                        len(records) for records in self.ground_archive.values()
                    ),
                },
            ),
            tasks=tasks,
            events=obs.events,
        )

    @property
    def delivered_object_ids(self) -> set[str]:
        return {object_id for object_id, records in self.ground_archive.items() if records}

    @property
    def onboard_coverage(self) -> float:
        if self.target_count <= 0:
            return 0.0
        known = {
            target_id
            for sat_estimates in self.onboard_estimates.values()
            for target_id in sat_estimates
        }
        return len(known) / self.target_count

    @property
    def delivered_coverage(self) -> float:
        return len(self.delivered_object_ids) / self.target_count if self.target_count else 0.0

    def get_metrics(self) -> Dict[str, float]:
        return {
            **super().get_metrics(),
            **self._ssa_metrics(),
        }

    def _decode_actions(self, actions: Dict[str, Any]) -> dict[str, str]:
        decoded: dict[str, str] = {}
        for sat_id in self._sat_ids:
            action = actions.get(sat_id, {}) if isinstance(actions, dict) else {}
            mode = "charging"
            if isinstance(action, dict):
                raw_mode = action.get("mode")
                if raw_mode in self.mode_to_index:
                    mode = str(raw_mode)
                elif "mode_one_hot" in action:
                    mode = self._decode_one_hot(action.get("mode_one_hot"))
                elif "one_hot" in action:
                    mode = self._decode_one_hot(action.get("one_hot"))
            elif isinstance(action, (list, tuple)):
                mode = self._decode_one_hot(action)
            decoded[sat_id] = mode
        return decoded

    def _decode_one_hot(self, values: Any) -> str:
        try:
            seq = [int(v) for v in values]
        except TypeError:
            return "charging"
        if len(seq) != len(self.mode_list):
            return "charging"
        if sum(1 for value in seq if value == 1) != 1:
            return "charging"
        return self.mode_list[seq.index(1)]

    def _record_detection(
        self, sat_id: str, access: DetectionAccess, *, step: int | None = None
    ) -> None:
        target_idx = self.target_index.get(access.object_id)
        if target_idx is None:
            return
        sat_idx = self._sat_index(sat_id)
        step = self.current_step if step is None else int(step)
        already_known = any(row[target_idx] for row in self.detection_matrix)
        # Re-detecting an object this satellite saw last step is a continued
        # track, not wasted duplicated effort.
        last_here = self._last_detection_by_sat.get((sat_id, access.object_id))
        continuation = last_here is not None and step - last_here <= 1
        if not already_known:
            self.successful_observations += 1
            self._first_detected_step[access.object_id] = step
        elif not continuation:
            self.duplicate_observations += 1
        self.total_observation_records += 1

        previous = self._last_observed_step.get(access.object_id)
        if previous is not None and step - previous > 1:
            self._revisit_intervals.append(step - previous)

        self.detection_matrix[sat_idx][target_idx] = 1
        record = {
            "object_id": access.object_id,
            "satellite_id": sat_id,
            "position_km": list(access.position_km),
            "time_step": step,
            "last_refresh_step": step,
            "m": float(access.m),
            "p_detect": float(access.p_detect),
            "quality": float(access.quality),
            "relay_hops": 0,
            "first_detected_step": self._first_detected_step.get(access.object_id, step),
        }
        current = self.onboard_estimates[sat_id].get(access.object_id)
        if current is None or float(record["quality"]) > float(current.get("quality", 0.0)):
            self.onboard_estimates[sat_id][access.object_id] = record
        else:
            current["last_refresh_step"] = step
        buffer = self._undelivered_records[sat_id]
        held = buffer.get(access.object_id)
        if held is None or float(record["quality"]) >= float(held.get("quality", 0.0)):
            buffer[access.object_id] = record
        self._last_observed_step[access.object_id] = step
        self._last_detection_by_sat[(sat_id, access.object_id)] = step
        self.last_step_detections.setdefault(sat_id, []).append(access.object_id)

    def _apply_isl_shares(self, modes: Mapping[str, str]) -> dict[str, float]:
        sharers = [sat_id for sat_id, mode in modes.items() if mode == "isl_share"]
        if not sharers:
            return {}
        energy_by_sat: dict[str, float] = {}
        t1 = self.current_step * self.step_duration_s
        t0 = t1 - self.step_duration_s
        for src_id in sharers:
            if self._subenvs[src_id].battery_soc < self.isl_min_soc:
                continue
            energy_by_sat[src_id] = self._bill_isl_power(src_id)
            feasible: dict[str, float] = {}
            for dst_id in self._sat_ids:
                if dst_id == src_id:
                    continue
                self.isl_attempts += 1
                dst_idle = modes.get(dst_id, "charging") in {"charging", "safe", "isl_share"}
                if not dst_idle:
                    continue
                capacity = self._isl_window_capacity_bytes(src_id, dst_id, t0, t1)
                if capacity <= 0.0:
                    continue
                self.isl_successes += 1
                feasible[dst_id] = capacity
                self._merge_satellite_knowledge(src_id, dst_id)
            if not (self.isl_relay and feasible):
                continue
            if self.isl_unicast:
                best = max(feasible, key=lambda dst: feasible[dst])
                targets = [(best, feasible[best])]
            else:
                targets = list(feasible.items())
            for dst_id, capacity in targets:
                self._relay_records(src_id, dst_id, capacity)
        return energy_by_sat

    def _isl_window_capacity_bytes(
        self, src_id: str, dst_id: str, t0: float, t1: float
    ) -> float:
        """Bytes transferable in [t0, t1] — the link budget integrated over
        sub-minute feasibility samples, mirroring the second-accurate handling
        of ground-pass windows."""
        resolution = max(1.0, self.isl_substep_s)
        capacity = 0.0
        t = t0
        while t < t1 - 1e-9:
            dt = min(resolution, t1 - t)
            midpoint = t + dt / 2.0
            src_pos = self._satellite_position(src_id, midpoint)
            dst_pos = self._satellite_position(dst_id, midpoint)
            distance_m = max(1.0, vector_range_km(src_pos, dst_pos) * 1000.0)
            capacity += effective_data_rate_bps(distance_m, self.isl_config) * dt / 8.0
            t += dt
        return capacity

    def _relay_records(self, src_id: str, dst_id: str, capacity_bytes: float) -> None:
        buffer = self._undelivered_records[src_id]
        if not buffer:
            return
        budget = capacity_bytes
        for object_id in sorted(buffer, key=lambda oid: buffer[oid]["time_step"]):
            if budget < self.record_size_bytes:
                break
            record = dict(buffer.pop(object_id))
            budget -= self.record_size_bytes
            record["relay_hops"] = int(record.get("relay_hops", 0)) + 1
            self.isl_records_relayed += 1
            self.isl_bytes_transferred += self.record_size_bytes
            dst_buffer = self._undelivered_records[dst_id]
            held = dst_buffer.get(object_id)
            if held is None or float(record["quality"]) > float(held.get("quality", 0.0)):
                dst_buffer[object_id] = record

    def _bill_isl_power(self, sat_id: str) -> float:
        # isl_share runs as `charging` inside the sub-env (bus attitude is
        # unconstrained for a UHF broadcast); the radio's extra draw is billed
        # here so sharing is never energetically free.
        sub = self._subenvs[sat_id]
        # Capacity belongs to the satellite whose SOC is being debited.  Using
        # the aggregate environment's prototype capacity would mis-account a
        # heterogeneous constellation.
        configured_capacity_wh = (
            (getattr(sub, "scenario", {}) or {})
            .get("power", {})
            .get("battery", {})
            .get("capacity_wh", 84.0)
        )
        capacity_wh = float(
            getattr(sub, "battery_capacity_wh", None)
            or configured_capacity_wh
            or 84.0
        )
        before_soc = float(sub.battery_soc)
        delta_soc = self.isl_power_overhead_w * (self.step_duration_s / 3600.0) / capacity_wh
        sub.battery_soc = max(0.0, sub.battery_soc - delta_soc)
        return (before_soc - float(sub.battery_soc)) * capacity_wh

    def _merge_satellite_knowledge(self, src_id: str, dst_id: str) -> None:
        src_idx = self._sat_index(src_id)
        dst_idx = self._sat_index(dst_id)
        for target_id, src_record in self.onboard_estimates.get(src_id, {}).items():
            target_idx = self.target_index[target_id]
            self.detection_matrix[dst_idx][target_idx] = max(
                self.detection_matrix[dst_idx][target_idx],
                self.detection_matrix[src_idx][target_idx],
            )
            dst_record = self.onboard_estimates[dst_id].get(target_id)
            refresh_step = max(0, self.current_step - 1)
            if dst_record is None or float(src_record.get("quality", 0.0)) > float(dst_record.get("quality", 0.0)):
                merged = deepcopy(src_record)
                merged["last_refresh_step"] = refresh_step
                self.onboard_estimates[dst_id][target_id] = merged
            else:
                dst_record["last_refresh_step"] = refresh_step

    def _apply_ground_downlinks(self, modes: Mapping[str, str], per_sat_info: Mapping[str, Any]) -> None:
        # Records are kB-scale track messages; a single S-band pass moves
        # hundreds, so record delivery is gated by pass access, not rate.
        for sat_id, mode in modes.items():
            if mode != "communication":
                continue
            sat_info = per_sat_info.get(sat_id, {})
            # Requested communication is not a physical outcome: ADCS settling
            # and safety can resolve it to charging/safe.  Archive records only
            # after productive communication actually resolved during positive
            # second-accurate physical contact.  Failed requests leave custody
            # buffers untouched.  Archived pre-fix SSA results are not
            # comparable because they credited such failed uploads.
            if sat_info.get("resolved_mode") != "communication":
                continue
            if bool(sat_info.get("in_transition", False)):
                continue
            if not bool(sat_info.get("physical_ground_pass_active", False)):
                continue
            try:
                contact_seconds = float(sat_info.get("contact_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                contact_seconds = 0.0
            if contact_seconds <= 0.0:
                continue
            buffer = self._undelivered_records.get(sat_id, {})
            for object_id, record in buffer.items():
                newly_delivered = not self.ground_archive.get(object_id)
                self.ground_archive.setdefault(object_id, []).append(deepcopy(record))
                if newly_delivered:
                    self._delivered_step[object_id] = self.current_step
                    if int(record.get("relay_hops", 0)) > 0:
                        self._relayed_first_deliveries += 1
            self.last_step_downlinked_records += len(buffer)
            self._undelivered_records[sat_id] = {}

    def _ssa_info(self, modes: Mapping[str, str]) -> dict[str, Any]:
        metrics = self._ssa_metrics()
        return {
            **metrics,
            "ssa_requested_modes": dict(modes),
            "ssa_detection_matrix": deepcopy(self.detection_matrix),
            "ssa_last_step_detections": deepcopy(self.last_step_detections),
            "ssa_step_downlinked_records": float(self.last_step_downlinked_records),
        }

    def _ssa_metrics(self) -> dict[str, float]:
        duplicate_rate = (
            self.duplicate_observations / self.total_observation_records
            if self.total_observation_records else 0.0
        )
        mean_revisit = (
            sum(self._revisit_intervals) / len(self._revisit_intervals)
            if self._revisit_intervals else 0.0
        )
        staleness_ages = [
            self.current_step - step for step in self._last_observed_step.values()
        ]
        mean_staleness = sum(staleness_ages) / len(staleness_ages) if staleness_ages else 0.0
        connectivity = self.isl_successes / self.isl_attempts if self.isl_attempts else 0.0
        known_objects = {
            target_id
            for sat_estimates in self.onboard_estimates.values()
            for target_id in sat_estimates
        }
        delivered = self.delivered_object_ids
        delivery_latencies = [
            self._delivered_step[oid] - self._first_detected_step[oid]
            for oid in delivered
            if oid in self._delivered_step and oid in self._first_detected_step
        ]
        mean_delivery_latency = (
            sum(delivery_latencies) / len(delivery_latencies)
            if delivery_latencies else 0.0
        )
        relayed_fraction = (
            self._relayed_first_deliveries / len(delivered) if delivered else 0.0
        )
        coverage_auc = self._coverage_auc_sum / max(1, self.current_step)
        return {
            "ssa_onboard_coverage": self.onboard_coverage,
            "ssa_delivered_coverage": self.delivered_coverage,
            "ssa_delivered_coverage_auc": coverage_auc,
            "ssa_known_objects": float(len(known_objects)),
            "ssa_delivered_objects": float(len(delivered)),
            "physical_utility_ceiling": float(self.physical_utility_ceiling),
            "successful_observations": float(self.successful_observations),
            "duplicate_observations": float(self.duplicate_observations),
            "duplicate_observation_rate": duplicate_rate,
            "mean_revisit_steps": mean_revisit,
            "mean_staleness_steps": mean_staleness,
            "mean_delivery_latency_steps": mean_delivery_latency,
            "relayed_delivery_fraction": relayed_fraction,
            "isl_connectivity": connectivity,
            "isl_records_relayed": float(self.isl_records_relayed),
            "isl_bytes_transferred": float(self.isl_bytes_transferred),
        }

    def _target_positions(self) -> dict[str, tuple[float, float, float]]:
        return self._target_positions_at(self.current_step * self.step_duration_s)

    def _target_positions_at(
        self, epoch_seconds: float
    ) -> dict[str, tuple[float, float, float]]:
        if self.fixed_target_positions:
            return dict(self.fixed_target_positions)
        return propagated_catalog_positions_km(
            self.targets,
            epoch_seconds,
            prefer_orekit=self.prefer_orekit_targets,
        )

    def _sun_hat_at(self, epoch_seconds: float) -> tuple[float, float, float]:
        return sun_unit_eci(epoch_seconds, _ELEMENT_EPOCH)

    def _observer_velocity_unit(
        self,
        sat_id: str,
        epoch_seconds: float,
    ) -> tuple[float, float, float]:
        position = self._satellite_position(sat_id, epoch_seconds)
        future = self._satellite_position(sat_id, epoch_seconds + 1.0)
        velocity = tuple(b - a for a, b in zip(position, future))
        norm = math.sqrt(sum(value * value for value in velocity))
        if norm > 1e-12:
            return tuple(value / norm for value in velocity)

        # Static-position fixtures have no propagated velocity. Give them a
        # deterministic local tangent so they remain useful without weakening
        # the real finite-difference contract.
        radius = math.sqrt(sum(value * value for value in position))
        if radius <= 0.0:
            return (1.0, 0.0, 0.0)
        radial = tuple(value / radius for value in position)
        reference = (0.0, 0.0, 1.0) if abs(radial[2]) < 0.9 else (1.0, 0.0, 0.0)
        tangent = (
            radial[1] * reference[2] - radial[2] * reference[1],
            radial[2] * reference[0] - radial[0] * reference[2],
            radial[0] * reference[1] - radial[1] * reference[0],
        )
        tangent_norm = math.sqrt(sum(value * value for value in tangent))
        return tuple(value / tangent_norm for value in tangent)

    def _optical_accesses_at(
        self,
        sat_id: str,
        epoch_seconds: float,
        *,
        target_positions: Mapping[str, Any] | None = None,
        restrict_ids: set[str] | None = None,
    ) -> list[DetectionAccess]:
        positions = (
            self._target_positions_at(epoch_seconds)
            if target_positions is None
            else dict(target_positions)
        )
        if restrict_ids is not None:
            positions = {
                object_id: position
                for object_id, position in positions.items()
                if object_id in restrict_ids
            }
        if not positions:
            return []
        return optical_accesses(
            self._satellite_position(sat_id, epoch_seconds),
            self._observer_velocity_unit(sat_id, epoch_seconds),
            self.targets,
            positions,
            self._sun_hat_at(epoch_seconds),
            fov_half_angle_deg=self.fov_half_angle_deg,
            boresight_pitch_deg=self.boresight_pitch_deg,
            r_cap_km=self.r_cap_km,
            m_lim=self.m_lim,
            sigma_m=self.sigma_m,
            albedo=self.albedo,
        )

    def _parse_fixed_target_positions(self, raw: Any) -> dict[str, tuple[float, float, float]]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return {str(k): tuple(float(x) for x in v) for k, v in raw.items()}
        out = {}
        for idx, value in enumerate(raw):
            out[f"rso_{idx}"] = tuple(float(x) for x in value)
        return out

    def _satellite_position(
        self, sat_id: str, epoch_seconds: float | None = None
    ) -> tuple[float, float, float]:
        if sat_id in self._configured_sat_positions:
            return self._configured_sat_positions[sat_id]
        orbit = self._sat_orbits.get(sat_id)
        if orbit is not None:
            t = (
                self.current_step * self.step_duration_s
                if epoch_seconds is None
                else float(epoch_seconds)
            )
            return propagate_rso_position_km(
                orbit, t, prefer_orekit=self.prefer_orekit_positions
            )
        return (0.0, 0.0, 500.0)

    def _sat_index(self, sat_id: str) -> int:
        try:
            return self._sat_ids.index(sat_id)
        except ValueError as exc:
            raise KeyError(f"unknown satellite id {sat_id!r}") from exc
