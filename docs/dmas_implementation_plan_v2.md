# DMAS completion plan V2

> This is the concise, authoritative implementation plan for the first complete
> SSA DMAS increment. It supersedes `dmas_implementation_plan.md` for
> implementation decisions; the older file remains only as planning history.
> This document does not itself authorise or implement source changes. After the
> increment lands, the durable design must be folded into the existing canonical
> documentation and both temporary plans may be removed.

## 1. Goal

Complete SSA DMAS as a genuinely decentralised organisation:

- one equal reasoning agent per satellite;
- each agent observes and commands only its own satellite;
- agents are logical all-to-all peers;
- knowledge learned by one satellite reaches another only through an actual
  `isl_share`;
- received knowledge becomes part of the receiver's local state for its next
  decision;
- local actions are composed because their satellite ownership is disjoint, not
  through plurality or full-plan consensus.

Kim et al. (2025) [FVFQ73RF] is the basis for the decentralised peer topology,
not for the SSA payload or a textual debate protocol. The payload and physical
gates remain grounded in the existing SSA mission/link implementation (Oliver
et al. [ACMHUJ8C] and the current Pachler-based link-budget model [7PTYIMJL]).

## 2. Scope rule

Build general contracts only where DMAS must connect to the framework. Activate
the new organisational behaviour only for DMAS.

This increment may:

- extend generic organisation/environment interfaces;
- correct generic copying, source attribution, and validation needed by local
  views;
- update the shared SSA RL observation schema;
- make the minimal symbolic adaptation required to recognise an ISL peer.

It must not change the organisational semantics of SAS, IMAS, HMAS, or CMAS.
Their completion or reinterpretation belongs to later increments. In
particular, CMAS keeps its current explicit unsupported/fail-fast cases.

It also must not change EventSat mission logic, LLM implementations or prompts,
reward semantics, or the existing symbolic rule order and thresholds.

The shared SSA RL block is the one deliberate cross-organisation schema change:
all SSA organisations reuse the same local per-satellite features, composed
according to their existing scopes. This versions/rejects old SSA checkpoints
but does not change organisation topology, ownership, or aggregation.

## 3. Non-negotiable invariants

The implementation does **not** introduce a communication-failure model.

- No stochastic packet loss.
- No sampled BER success/failure. BER remains part of the existing
  deterministic effective-rate calculation.
- No ACK, response assumption, retry, timeout, retransmission, or
  `delivered_version`.
- No per-peer delivery history.
- No new battery, receiver, capacity, or energy gate.

The existing SSA feasibility rules remain authoritative:

- the source must satisfy the existing ISL minimum SoC;
- the receiver must be idle under the existing mode rule;
- integrated link capacity must be positive;
- an admitted transmitter pays the existing radio-energy cost once per step.

If those existing conditions admit a link, knowledge delivery is
deterministic. A link excluded by a gate is physically unavailable for that
step, not a randomly failed message.

Other invariants:

- do not reintroduce `visible_rso_ids` or another perfect current-FOV oracle;
- preserve the current record-relay, capacity, unicast, custody, and
  `ssa.isl_relay` semantics;
- preserve the current ground-downlink and reward behaviour;
- do not use `AgentObservation.messages` as a second SSA data plane.

## 4. Target step semantics

For a decision step `t`:

1. The environment exposes its full internal constellation observation.
2. DMAS projects one strict local view for each peer.
3. Each representation receives its satellite's physical state and accumulated
   local SSA knowledge, including knowledge delivered in earlier steps.
4. Each peer independently emits an action only for its own satellite.
5. DMAS validates ownership and merges the disjoint actions.
6. The environment evaluates all actions simultaneously.
7. Every admitted `isl_share` source sends a snapshot of its local knowledge to
   every authorised peer whose existing physical gates are satisfied.
8. The existing capacity-limited custody relay is planned separately from the
   knowledge broadcast and then committed.
9. The next observation contains the merged receiver state, so it can affect
   decisions at `t+1`.

All sources, destinations, knowledge, and custody buffers are snapshotted before
any receiver is mutated. Therefore `A -> B -> C` cannot occur in one step.
Knowledge received by B may legitimately be forwarded by B in a later step.

## 5. Information visible to a DMAS peer

`sat_agent_i` receives:

- the physical/resource/pipeline state of `sat_i`;
- its own detection row;
- its own known-object list and copied best estimates;
- its own knowledge ages and known-object prediction cues;
- its own pending custody-record count;
- whether it has at least one authorised outgoing ISL peer;
- tasks and events explicitly attributable to `sat_i`;
- timestep/epoch and configuration constants already supplied through the
  representation configuration.

It does not receive:

- another satellite state;
- the full detection matrix;
- team onboard/delivered coverage;
- the ground archive or global delivered-object set;
- another peer's current or previous action;
- another peer's record buffer;
- unaddressed tasks/events whose source cannot be established;
- instantaneous link feasibility or an implicit receiver reply.

Knowledge received through earlier ISL actions is local knowledge from that
point onward. It is therefore correctly included in subsequent observations;
locality does not mean “self-acquired facts only”.

### Observation-schema boundary

Use the existing distinction between per-satellite metadata and
`ConstellationState.global_info` as a real contract:

- dynamic satellite-local SSA facts remain in `SatelliteState.metadata`;
- global diagnostic truth remains in the full observation/global info and
  `StepResult.info`;
- duplicated global matrix/coverage/archive fields are not exposed inside a
  strict DMAS satellite view.

Global organisations must retain access to the same global truth through the
canonical global container. Any relocation needed to remove duplicated fields
is a schema adaptation, not a loss of their information.

Extend `scope_observation()` only with generic opt-in controls:

- copied satellite states/metadata;
- include or omit `global_info`;
- permissive or strict filtering of addressed tasks/events.

Defaults preserve current non-DMAS behaviour. DMAS opts into copied strict
local views with no dynamic global info.

`MultiEventsatEnv` must tag aggregated tasks/events with their source satellite
when that source is known. This is attribution only; it must not change task or
event generation.

## 6. Minimal general framework contracts

### 6.1 Logical topology

Add one optional organisation contract, conceptually:

```python
def logical_communication_edges(self) -> set[tuple[str, str]] | None:
    return None
```

Semantics:

- `None`: the organisation has not opted into authoritative transport binding;
  preserve the environment's current behaviour;
- `set()`: an explicitly declared topology with no peer links;
- non-empty set: authorised directed agent-to-agent links.

DMAS returns every directed pair of distinct peers. The generic composition
layer maps those agent edges through `satellites_for_agent()` to physical
satellite endpoints and validates all IDs.

Only DMAS opts into this contract in this increment. Do not declare or infer
new CMAS, IMAS, HMAS, or SAS communication behaviour now.

### 6.2 Environment binding

Add a scenario-independent optional environment capability, conceptually:

```python
def configure_communication_links(
    self,
    links: set[tuple[str, str]] | None,
) -> None:
    ...
```

The base implementation is a no-op. SSA stores the binding and uses it only to
restrict candidate ISL destinations:

- `None` preserves the current legacy candidate set;
- an empty set authorises no endpoint;
- a non-empty set authorises exactly those endpoints.

Binding does not say that a physical link is currently feasible and does not
add a failure mechanism.

Use one shared binding helper in both the direct runner and RLlib. Bind after
environment reset and organisation initialisation, then refresh the first
observation so `has_isl_peer` is correct before the first decision.

### 6.3 Action ownership

Keep static action-scope validation in
`validate_agent_satellite_mapping()`. In DMAS `collect_actions()` additionally:

- reject an action for a satellite outside the producing peer's action scope;
- reject overlapping satellite keys;
- merge valid disjoint local actions directly.

Do not add plurality, serialisation, a full-plan fallback, or a new failure for
an omitted per-step action; the environment's existing default handling remains.
Because both execution paths call the same organisation method, this also gives
direct-runner/RLlib parity without a second RL-only rule.

## 7. DMAS implementation changes

In `DecentralizedMAS`:

- `observed_satellites_for_agent()` returns only the peer's own satellite;
- `satellites_for_agent()` remains the same singleton action scope;
- `logical_communication_edges()` returns the all-to-all directed peer graph;
- `distribute_observation()` returns strict copied local views;
- `AgentObservation.messages` remains empty;
- `collect_actions()` performs ownership validation and disjoint merge;
- remove `_last_round_messages`, JSON/`Counter` plurality, consensus fallback,
  `coordination_messages`, and `consensus_rounds`.

Do not replace those synthetic counts with another estimated message counter.
Physical communication is already measured by SSA's actual ISL metrics.

## 8. SSA transport changes

Refactor `_apply_isl_shares()` into plan and commit phases while retaining all
existing gates and accounting:

1. collect admitted `isl_share` sources;
2. restrict destinations to authorised links when a topology is bound;
3. apply the existing receiver-idle and capacity checks;
4. bill the existing per-source energy;
5. snapshot every source's detection row, best estimates, and custody buffer;
6. plan deterministic knowledge deliveries to all feasible destinations;
7. plan record-custody routes with the current unicast/capacity rules;
8. merge knowledge from snapshots;
9. commit each custody movement at most once;
10. update the existing physical ISL metrics.

The knowledge payload is:

```text
sender's complete accumulated detection row
+ sender's locally retained best estimate for each known object
+ existing acquisition/provenance fields
```

Rows merge with OR. Estimates use one deterministic ordering: better quality,
then newer acquisition step, then stable satellite/object tie-break. Relaying
must not rewrite acquisition time, and receiving the same or an inferior
estimate must not falsely refresh its age.

Knowledge broadcast and record custody remain distinct:

- knowledge reaches every feasible authorised receiver;
- custody follows the existing capacity-limited relay/unicast policy;
- a custody record moves rather than being copied;
- `ssa.isl_relay: false` disables custody movement, not knowledge dissemination.

Preserve the meanings of existing `isl_attempts`, `isl_successes`,
`isl_records_relayed`, `isl_bytes_transferred`, and ISL energy. Do not add a
reliability metric for a stochastic process that does not exist.

## 9. Representation adaptation

### 9.1 Shared SSA RL block

Replace the current 32D block with one bounded local SSA block shared by the
scenario representation and RLlib adapter. Freeze it under a schema identifier
such as `ssa_local_compact_v1`.

The 30 features are:

- 8 resource/health features: battery, OBC fraction, total stored fraction,
  Jetson raw, Jetson compressed, contact, health, sunlight;
- 5 local-knowledge/communication features: own-row known fraction, pending
  record fraction, predicted-known-in-FOV fraction, mean known-estimate age,
  `has_isl_peer`;
- 9 timing/pipeline features: remaining pass, next pass, next eclipse, episode
  phase, raw backlog, detection backlog, compression progress, detection
  progress, downlink utilisation;
- 8 current-mode one-hot features.

Remove the absent visibility-oracle features, global delivered/coverage
features, and previous-action message vector. Remove
`include_peer_messages: true` from DMAS SSA configs.

Organisations compose the same local blocks according to their existing
observation scopes:

- DMAS receives one block;
- a global or clustered agent may concatenate multiple blocks.

This shared SSA encoder change is intentional and representation-independent;
EventSat encoders remain untouched. Version the schema and reject old
incompatible SSA checkpoints with a clear schema/dimension error rather than a
late tensor-shape failure.

### 9.2 Symbolic adaptation

Expose `has_isl_peer` in the structured local state and use:

```text
coordinated = controls_multiple_satellites or has_isl_peer
```

This is the only intended symbolic logic adaptation. Do not change rule
priority, thresholds, ground communication, backpressure, or relay conditions.
The separate no-oracle symbolic discovery-policy redesign remains deferred.

### 9.3 Other representations

Do not modify LLM implementations or prompts. The local observation and
topology contracts are representation-agnostic, so a future symbolic, RL, or
LLM DMAS agent can consume the same framework view without a second physical
message path.

## 10. Verification

Add focused tests proving:

- every DMAS peer sees exactly one copied satellite and no dynamic global truth;
- previously received ISL knowledge is present in the receiver's next local
  observation;
- tasks/events are source-filtered and views cannot mutate one another;
- DMAS actions cannot command foreign satellites and valid local actions merge;
- the declared DMAS graph and authorised endpoint graph are all-to-all;
- direct runner and RLlib bind the same topology after reset;
- unbound non-DMAS environments preserve existing candidate-link behaviour;
- existing deterministic SoC, receiver-idle, capacity, energy, unicast, relay,
  and downlink rules are unchanged;
- one `isl_share` delivers knowledge to every feasible authorised peer;
- no same-step `A -> B -> C` knowledge or custody cascade occurs;
- repeated/inferior estimates do not become artificially fresh;
- custody records are not duplicated;
- the local RL block is exactly 30D, bounded, and identical in both encoders;
- no oracle, global coverage, delivered set, or peer-action vector enters the
  DMAS RL observation;
- the symbolic policy can consider `isl_share` through `has_isl_peer` without
  changing its existing rule precedence;
- SAS, IMAS, HMAS, and CMAS retain their current scopes, aggregation, and
  support/fail-fast status; EventSat and LLM-related behaviour remains
  unchanged.

Run focused organisation, SSA, RL adapter, runner, RLlib, symbolic, and config
tests first, then:

```powershell
uv run pytest tests/ -v
```

## 11. Implementation order

1. Add failing tests for strict local views, DMAS ownership, and topology
   binding.
2. Add the optional generic topology/binding and scoping contracts, preserving
   all defaults.
3. Replace DMAS global/plurality behaviour with local views and disjoint action
   composition.
4. Refactor SSA ISL into snapshot plan/commit and expose local knowledge plus
   `has_isl_peer`.
5. Freeze the 30D local SSA schema, remove DMAS peer-action vectors, and make
   the minimal symbolic adaptation.
6. Run focused tests and symbolic/RL-mock DMAS smokes at N=3 and N=5.
7. Update `docs/implementations.md`, `docs/morphological_matrix.md`,
   `docs/architecture.md`, and `docs/scenarios.md`; run the full suite.

## 12. Explicitly deferred

- completion or redesign of CMAS, HMAS, IMAS, or SAS communication semantics;
- Kim-style textual debate, multi-round proposal consensus, or target claims;
- recipient-selection actions or learned/compressed message payloads;
- ACKs, replies, retry queues, `delivered_version`, timeouts, or stochastic
  packet loss;
- topology ablations beyond the first all-to-all DMAS;
- LLM inbox/prompt integration;
- the no-oracle symbolic discovery-policy rewrite;
- native non-AO SSA scheduling;
- target-wise variable-length RL encoders.
