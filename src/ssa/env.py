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

from copy import deepcopy
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
from src.ssa.rewards import SSARewardFunction
from src.ssa.targets import (
    _EARTH_RADIUS_KM as EARTH_RADIUS_KM,
    DetectionAccess,
    RSOTarget,
    detect_targets_in_fov,
    generate_sso_catalog,
    propagate_rso_position_km,
    propagated_catalog_positions_km,
)


SSA_MODES = [
    "charging",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
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

        self.reward_fn = SSARewardFunction(config.get("reward_config", {}))
        self.mode_list = list(SSA_MODES)
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.mode_list)}
        self.target_count = int(self.targets_config.get("count", 6))
        self.fov_half_angle_deg = float(self.targets_config.get("fov_half_angle_deg", 5.0))
        self.max_detection_range_km = self.targets_config.get("max_range_km")
        if self.max_detection_range_km is not None:
            self.max_detection_range_km = float(self.max_detection_range_km)
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
            self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}
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
        return self.get_observation()

    def _sat_orbit_elements(self, sat_id: str) -> RSOTarget:
        orbit = getattr(self._subenvs[sat_id], "_episode_orbit", None) or {}
        return RSOTarget(
            object_id=sat_id,
            semi_major_axis_km=EARTH_RADIUS_KM + float(orbit.get("altitude_km", 400.0)),
            eccentricity=float(orbit.get("eccentricity", 0.001)),
            inclination_deg=float(orbit.get("inclination_deg", 97.4)),
            raan_deg=float(orbit.get("raan_deg", 0.0)),
            arg_perigee_deg=float(orbit.get("arg_perigee_deg", 0.0)),
            true_anomaly_deg=float(orbit.get("true_anomaly_deg", 0.0)),
            epoch=_ELEMENT_EPOCH,
        )

    def step(self, actions: Dict[str, Any]) -> StepResult:
        decoded = self._decode_actions(actions)
        event_actions = {
            sat_id: {"mode": ("charging" if mode == "isl_share" else mode)}
            for sat_id, mode in decoded.items()
        }
        base_result = super().step(event_actions)
        per_sat_info = dict(base_result.info.get("per_satellite", {}))

        self.last_step_detections = {sat_id: [] for sat_id in self._sat_ids}
        self.last_step_downlinked_records = 0
        target_positions = self._target_positions()

        for sat_id, requested_mode in decoded.items():
            info = per_sat_info.get(sat_id, {})
            productive_observe = (
                requested_mode == "payload_observe"
                and info.get("resolved_mode") == "payload_observe"
                and not bool(info.get("in_transition", False))
            )
            if productive_observe:
                accesses = detect_targets_in_fov(
                    self._satellite_position(sat_id),
                    target_positions,
                    fov_half_angle_deg=self.fov_half_angle_deg,
                    max_range_km=self.max_detection_range_km,
                )
                for access in accesses:
                    self._record_detection(sat_id, access)

        self._apply_isl_shares(decoded)
        self._apply_ground_downlinks(decoded, per_sat_info)
        self._coverage_auc_sum += self.delivered_coverage

        info = dict(base_result.info)
        info.update(self._ssa_info(decoded))
        rewards = self.reward_fn.compute_rewards(
            base_result.rewards,
            {"_global": {"delivered_coverage": self.delivered_coverage}},
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
        tasks = list(obs.tasks)
        for sat_id, sat in obs.constellation_state.satellites.items():
            position = self._satellite_position(sat_id)
            visible = detect_targets_in_fov(
                position,
                target_positions,
                fov_half_angle_deg=self.fov_half_angle_deg,
                max_range_km=self.max_detection_range_km,
            ) if target_positions else []
            metadata = dict(sat.metadata)
            sat_idx = self._sat_index(sat_id)
            metadata.update({
                "ssa_detection_matrix": deepcopy(self.detection_matrix),
                "ssa_detection_row": list(self.detection_matrix[sat_idx]) if self.detection_matrix else [],
                "ssa_known_objects": sorted(self.onboard_estimates.get(sat_id, {})),
                "ssa_delivered_objects": sorted(self.delivered_object_ids),
                "ssa_undelivered_records": len(self._undelivered_records.get(sat_id, {})),
                "visible_rso_ids": [item.object_id for item in visible],
                "visible_rso_count": len(visible),
                "ssa_onboard_coverage": self.onboard_coverage,
                "ssa_delivered_coverage": self.delivered_coverage,
            })
            satellites[sat_id] = SatelliteState(
                satellite_id=sat_id,
                position=list(position),
                velocity=list(sat.velocity),
                resources=dict(sat.resources),
                status=sat.status,
                metadata=metadata,
            )
            for access in visible:
                tasks.append({
                    "type": "observe_rso",
                    "satellite_id": sat_id,
                    "object_id": access.object_id,
                    "quality": access.quality,
                })

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
                    "ssa_ground_archive_records": sum(len(v) for v in self.ground_archive.values()),
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

    def _record_detection(self, sat_id: str, access: DetectionAccess) -> None:
        target_idx = self.target_index.get(access.object_id)
        if target_idx is None:
            return
        sat_idx = self._sat_index(sat_id)
        step = self.current_step
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
            "quality": float(access.quality),
            "relay_hops": 0,
            "first_detected_step": self._first_detected_step.get(access.object_id, step),
        }
        current = self.onboard_estimates[sat_id].get(access.object_id)
        if current is None or float(record["quality"]) > float(current.get("quality", 0.0)):
            self.onboard_estimates[sat_id][access.object_id] = record
        buffer = self._undelivered_records[sat_id]
        held = buffer.get(access.object_id)
        if held is None or float(record["quality"]) >= float(held.get("quality", 0.0)):
            buffer[access.object_id] = record
        self._last_observed_step[access.object_id] = step
        self._last_detection_by_sat[(sat_id, access.object_id)] = step
        self.last_step_detections.setdefault(sat_id, []).append(access.object_id)

    def _apply_isl_shares(self, modes: Mapping[str, str]) -> None:
        sharers = [sat_id for sat_id, mode in modes.items() if mode == "isl_share"]
        if not sharers:
            return
        t1 = self.current_step * self.step_duration_s
        t0 = t1 - self.step_duration_s
        for src_id in sharers:
            if self._subenvs[src_id].battery_soc < self.isl_min_soc:
                continue
            self._bill_isl_power(src_id)
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

    def _bill_isl_power(self, sat_id: str) -> None:
        # isl_share runs as `charging` inside the sub-env (bus attitude is
        # unconstrained for a UHF broadcast); the radio's extra draw is billed
        # here so sharing is never energetically free.
        sub = self._subenvs[sat_id]
        capacity_wh = float(getattr(self, "battery_capacity_wh", 84.0) or 84.0)
        delta_soc = self.isl_power_overhead_w * (self.step_duration_s / 3600.0) / capacity_wh
        sub.battery_soc = max(0.0, sub.battery_soc - delta_soc)

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
            if dst_record is None or float(src_record.get("quality", 0.0)) > float(dst_record.get("quality", 0.0)):
                self.onboard_estimates[dst_id][target_id] = deepcopy(src_record)

    def _apply_ground_downlinks(self, modes: Mapping[str, str], per_sat_info: Mapping[str, Any]) -> None:
        # Records are kB-scale track messages; a single S-band pass moves
        # hundreds, so record delivery is gated by pass access, not rate.
        for sat_id, mode in modes.items():
            if mode != "communication":
                continue
            pass_active = bool(per_sat_info.get(sat_id, {}).get("ground_pass_active", False))
            if not (pass_active or self.ground_always_visible):
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
        if self.fixed_target_positions:
            return dict(self.fixed_target_positions)
        return propagated_catalog_positions_km(
            self.targets,
            self.current_step * self.step_duration_s,
            prefer_orekit=self.prefer_orekit_targets,
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
