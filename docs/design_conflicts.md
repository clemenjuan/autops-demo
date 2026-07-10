---
status: internal-working-file
---

# Design Conflicts

**INTERNAL WORKING FILE — cross-substrate coordination. Not part of the
canonical documentation. Delete before publication.**

This file tracks design tensions that can affect comparisons across cognitive
substrates (symbolic, RL, LLM, hybrid). It is not a general decision log. Use it
only when one substrate makes a choice that deviates from the shared
organisation/representation/paradigm contract, or when a choice can distort
cross-substrate comparisons if other implementers do not see it.

Ground rule: scenarios are model-agnostic. Environments publish general
observations; substrate-specific contracts live in their own layer; the
environment is never modified to favour one substrate.

Each entry uses:

- `Date`
- `Area`
- `Status`: `Open`, `Acknowledged`, `Accepted`, or `Resolved`
- `What was decided`
- `Why`
- `Impact on comparisons`
- `Action needed from other substrates`

---

## DC-001 — DMAS Aggregation Differs Under the RL Substrate

**Date:** 2026-07-06  
**Area:** RL / organisation  
**Status:** Open

**What was decided.** The symbolic DMAS path keeps plurality consensus over full
constellation plans. The RL DMAS path uses the same all-to-all communication and
global observation scope, but each peer emits only the action for its own
satellite and the organisation merges those disjoint per-satellite proposals.

**Why.** PPO policies are stochastic during training. If each peer emitted a full
plan and `collect_actions` required byte-identical plans for plurality, almost
every proposal would be unique. Tie-breaking would repeatedly select one peer's
plan, and the other peers' proposed actions would have no causal effect on the
environment, breaking credit assignment.

**Impact on comparisons.** DMAS-symbolic and DMAS-RL do not share the exact same
orchestration policy (Omega). DMAS-symbolic guarantees deconfliction
structurally through the winning full plan; DMAS-RL exposes global information
and peer proposals, but deconfliction is learned rather than guaranteed. The
organisation-axis claim "DMAS avoids duplicate observations" must therefore be
read as structural for symbolic and as an experimental hypothesis for RL.

**Action needed from other substrates.** A future DMAS-LLM implementation must
declare whether each peer emits a full plan for plurality consensus or a
per-satellite action for merge. That choice determines whether DMAS-LLM is more
homogeneous with the symbolic or RL DMAS implementation.

---

## DC-002 — SAS Must Preserve a Single Decision Locus in Every Substrate

**Date:** 2026-07-06  
**Area:** All substrates / organisation  
**Status:** Accepted

**What was decided.** SAS implementations must execute as one reasoning locus
that observes the full constellation and emits actions for all satellites.
For RL, this means one joint policy with concatenated satellite observations and
joint factored actions, not centralized training with decentralized execution.

**Why.** CTDE-style execution would leave one runtime decision maker per
satellite, which is operationally close to IMAS with shared weights. That would
erase the intended organisation-axis contrast between SAS and IMAS.

**Impact on comparisons.** If any substrate implements SAS as multiple runtime
decision loci, SAS-vs-IMAS comparisons no longer isolate organisation.

**Action needed from other substrates.** Future LLM or hybrid SAS cells should
use one prompt/agentic loop that plans for all satellites, unless they explicitly
open a new design conflict.

---

## DC-003 — Multi-Satellite Agent Reward Reduction Equals Sum

**Date:** 2026-07-06  
**Area:** RL / reward interface  
**Status:** Accepted

**What was decided.** When one RL agent controls multiple satellites, its scalar
reward is the sum of the per-satellite rewards for those satellites.

**Why.** Summing preserves the total reward mass across organisations:
sum(agent rewards) equals sum(satellite rewards) regardless of how satellites are
grouped into agents. Averaging would shrink the SAS signal by roughly `N` versus
IMAS and would make PPO hyperparameters behave differently for reasons unrelated
to organisation.

**Impact on comparisons.** Episode-return comparisons remain on the same global
objective scale across SAS, HMAS, IMAS, and DMAS.

**Action needed from other substrates.** Any future learned substrate, shaping
term, normalization, or reward-derived metric must preserve this convention when
reducing per-satellite rewards to per-agent signals.
