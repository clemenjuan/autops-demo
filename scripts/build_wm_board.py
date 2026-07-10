"""Build the EventSat World-Model planner board (served separately on :8801).

A focused comparison surface for onboard planners, so the same CEM-MPC search can be
compared side by side over a learned latent rollout and the exact analytic EventSat
equations. Planner latency and energy are read from run artifacts, not inferred from
the configured power price.

Auto-discovers every WM-track run under ``data/results`` (by its representation type),
reads ``results.json`` for metrics and ``config.json`` for the planner knobs, and draws
a per-run SOC curve from ``decisions_ep0.jsonl`` when present. Static output →
``data/figures/wm_board/index.html``. Rebuilt by ``scripts/refresh_board.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("data/results")
OUT = Path("data/figures/wm_board/index.html")
FULL_WEEK_STEPS = 10080

WM_TYPES = {"lewm_cem_eventsat", "dreamerv3_eventsat"}
# Reference (non-WM) baselines shown greyed for context, keyed by result id.
REFERENCE_IDS = {
    "eventsat_sas_ao_symb": "Symbolic · rules on OBC — sub-watt AO reference",
}

# metric key -> (label, "higher"|"lower" better, format)
METRICS = [
    ("utility", "Utility", "higher", "{:.3f}"),
    (
        "utility_fraction_of_physical_ceiling",
        "Utility / physical ceiling",
        "higher",
        "{:.3f}",
    ),
    ("planner_ms_per_event", "Planner ms / event", "lower", "{:.2f}"),
    ("planner_energy_wh", "Planner Wh / episode", "lower", "{:.3f}"),
    ("artifact_loaded", "Artifact loaded", "neutral", "{:.0f}"),
    ("artifact_fallback", "Artifact fallback", "neutral", "{:.0f}"),
    ("downlinked_mb", "Downlink MB", "higher", "{:.2f}"),
    ("final_battery_soc", "Final SOC", "higher", "{:.3f}"),
    ("safety_overrides", "Safety ovr.", "lower", "{:.0f}"),
    ("observation_hours", "Obs. hrs", "neutral", "{:.2f}"),
]

_BACKENDS = {"latent", "analytic", "fallback"}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _selector(rc: dict, backend: str | None = None) -> tuple[str, str]:
    """Return (selector label, css class) for a representation config."""
    t = rc.get("type", "")
    if t == "dreamerv3_eventsat":
        return "DreamerV3 · model-free RL", "sel-rl"
    if t == "lewm_cem_eventsat":
        if backend == "analytic":
            return "CEM · exact analytic rollout", "sel-analytic"
        if backend == "fallback":
            return "CEM · deterministic fallback", "sel-fallback"
        sel = str(rc.get("selector", "cem")).lower()
        if sel in ("beam", "rl", "policy"):
            return "LeWM · Beam (RL top-k)", "sel-beam"
        return "LeWM · CEM (MPC)", "sel-cem"
    return t or "unknown", "sel-cem"


def _compute_power_w(cfg: dict, rc: dict | None = None) -> float:
    rc = rc or {}
    try:
        if rc.get("planner_power_w") is not None:
            return float(rc["planner_power_w"])
    except (TypeError, ValueError):
        pass
    scenario = ((cfg.get("environment") or {}).get("scenario_config") or {})
    overrides = scenario.get("scenario_overrides") or {}
    power = overrides.get("power") or {}
    try:
        return float(power.get("onboard_compute_w", 7.0))
    except (TypeError, ValueError):
        return 7.0


def _config_summary(
    rc: dict,
    compute_w: float | None = None,
    *,
    backend: str | None = None,
) -> str:
    t = rc.get("type", "")
    horizon = rc.get("horizon", "?")
    if t == "lewm_cem_eventsat":
        sel = str(rc.get("selector", "cem")).lower()
        hold = int(rc.get("plan_hold", 1) or 1)
        compute = 7.0 if compute_w is None else compute_w
        pricing = str(rc.get("planner_pricing", "jetson")).lower()
        price_label = "OBC-priced" if pricing == "obc" else "Jetson-priced"
        width = (
            f"beam={rc.get('beam_width', '?')}"
            if sel in ("beam", "rl", "policy")
            else f"{rc.get('samples', '?')}×{rc.get('cem_iterations', '?')}"
        )
        backend_label = backend or str(rc.get("planner_backend", "latent"))
        return (
            f"H={horizon} · hold={hold} · {backend_label} · {price_label} "
            f"Pplan={compute:g} W · {width} · {rc.get('mission_mode', '?')}"
        )
    return f"H={horizon} · heuristic-fallback"


def _jetson_duty(rc: dict) -> float | None:
    if rc.get("type") == "lewm_cem_eventsat":
        return 1.0 / max(1, int(rc.get("plan_hold", 1) or 1))
    if rc.get("type") == "dreamerv3_eventsat":
        return 1.0
    return None


def _flag_value(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _reported_backend(row: dict) -> str | None:
    """Return the backend recorded by the run, when the artifact supplies it."""
    value = row.get("reported_backend")
    if value is None:
        value = (row.get("mean") or {}).get("rollout_backend")
    value = str(value).strip().lower() if value is not None else ""
    return value if value in _BACKENDS else None


def _rollout_backend(row: dict, *, is_ref: bool = False) -> str:
    """Classify runtime rollout without conflating fallback and analytic CEM.

    An explicit fallback flag is the most conservative runtime truth. The named
    runtime backend wins next, then the config saved beside the result. Thus an
    artifact-load failure can never enter the intentional analytic group.
    """
    if is_ref or row.get("ref_label"):
        return "rule-based reference"
    rc = row.get("rc") or {}
    if rc.get("type") == "dreamerv3_eventsat":
        return "model-free"
    if rc.get("type") != "lewm_cem_eventsat":
        return "unknown"

    mean = row.get("mean") or {}
    fallback = _flag_value(mean.get("artifact_fallback"))
    if fallback is not None and fallback > 0.0:
        return "fallback"
    reported = _reported_backend(row)
    loaded = _flag_value(mean.get("artifact_loaded"))
    if reported is not None:
        if reported == "latent" and loaded is not None and loaded < 1.0:
            return "fallback"
        return reported
    intended = str(rc.get("planner_backend", "latent")).strip().lower()
    if intended == "analytic":
        return "analytic"
    if loaded is not None and loaded < 1.0:
        return "fallback"
    return "latent"


def _metric_value(row: dict, key: str) -> float | None:
    """Read canonical board metrics while retaining old result readability."""
    mean = row.get("mean") or {}
    aliases = {
        "planner_ms_per_event": (
            ("planner_ms_per_event", 1.0),
            ("planner_latency_ms_per_event", 1.0),
            ("planner_event_latency_ms", 1.0),
            # Legacy artifacts recorded seconds under this name.
            ("planner_latency_s", 1000.0),
        ),
        "planner_energy_wh": (
            ("planner_energy_wh", 1.0),
            ("planner_energy_wh_per_episode", 1.0),
            ("planner_energy_per_episode_wh", 1.0),
        ),
        "utility_fraction_of_physical_ceiling": (
            ("utility_fraction_of_physical_ceiling", 1.0),
            ("utility_fraction_of_ceiling", 1.0),
        ),
    }
    candidates = aliases.get(key, ((key, 1.0),))
    for candidate, scale in candidates:
        value = mean.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) * scale
    return None


def _row_status(row: dict, *, is_ref: bool = False) -> str:
    if not row.get("n"):
        return "notrun"
    backend = _rollout_backend(row, is_ref=is_ref)
    # Label fallback even for a smoke run; otherwise a short failed-artifact run
    # can visually resemble an intentional analytic smoke result.
    if backend == "fallback":
        return "fallback"
    steps = row.get("steps")
    if not isinstance(steps, int) or steps < FULL_WEEK_STEPS:
        return "smoke"
    if is_ref:
        return "measured"
    rc_type = (row.get("rc") or {}).get("type")
    mean = row.get("mean") or {}
    if rc_type == "lewm_cem_eventsat":
        intended = str((row.get("rc") or {}).get("planner_backend", "latent")).lower()
        reported = _reported_backend(row)
        if reported is not None and reported != intended:
            return "inconsistent"
        loaded = _flag_value(mean.get("artifact_loaded"))
        fallback = _flag_value(mean.get("artifact_fallback"))
        if backend == "analytic":
            if (loaded is not None and loaded > 0.0) or (
                fallback is not None and fallback > 0.0
            ):
                return "inconsistent"
        elif loaded is None and reported is None:
            return "unverified"
    if rc_type == "dreamerv3_eventsat":
        policy = _flag_value(mean.get("policy_loaded"))
        if policy is not None and policy < 1.0:
            return "fallback"
    return "measured"


def _soc_curve(rid: str, n_pts: int = 160) -> list[float]:
    f = RESULTS / rid / "decisions_ep0.jsonl"
    if not f.exists():
        return []
    socs: list[float] = []
    with f.open() as fh:
        for line in fh:
            try:
                socs.append(float(json.loads(line).get("battery_soc", 0.0)))
            except Exception:
                continue
    if len(socs) <= n_pts:
        return socs
    step = len(socs) / n_pts
    return [socs[min(len(socs) - 1, int(i * step))] for i in range(n_pts)]


def _sparkline(socs: list[float], w: int = 220, h: int = 40) -> str:
    if not socs:
        return '<span class="nodata">no trace</span>'
    n = len(socs)
    pts = " ".join(
        f"{(i/(n-1))*w:.1f},{h - max(0.0, min(1.0, s))*h:.1f}" for i, s in enumerate(socs)
    )
    # reserve line at 0.5 and safe floor at 0.2
    y_res = h - 0.5 * h
    y_safe = h - 0.2 * h
    end = socs[-1]
    stroke = "#1a7f37" if end > 0.4 else ("#d1242f" if end < 0.15 else "#bf8700")
    return (
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<line x1="0" y1="{y_res:.1f}" x2="{w}" y2="{y_res:.1f}" class="ref"/>'
        f'<line x1="0" y1="{y_safe:.1f}" x2="{w}" y2="{y_safe:.1f}" class="ref safe"/>'
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.6"/>'
        f"</svg>"
    )


def _result_backend(res: dict, mean: dict) -> str | None:
    """Find a runtime backend diagnostic across old and current artifact layouts."""
    statistics = res.get("experiment_statistics") or {}
    metadata = res.get("metadata") or {}
    candidates = (
        mean.get("rollout_backend"),
        statistics.get("rollout_backend"),
        res.get("rollout_backend"),
        metadata.get("rollout_backend"),
    )
    for value in candidates:
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized in _BACKENDS:
            return normalized
    return None


def collect() -> tuple[list[dict], list[dict]]:
    wm, ref = [], []
    for cfgp in sorted(RESULTS.glob("*/config.json")):
        rid = cfgp.parent.name
        cfg = _load(cfgp) or {}
        res = _load(cfgp.parent / "results.json") or {}
        rc = cfg.get("representation_config", {}) or {}
        mean = res.get("experiment_statistics", {}).get("mean", {}) or {}
        row = {
            "id": rid,
            "desc": (cfg.get("description") or "").strip(),
            "rc": rc,
            "compute_w": _compute_power_w(cfg, rc),
            "mean": mean,
            "reported_backend": _result_backend(res, mean),
            "n": len(res.get("episodes", [])),
            "steps": cfg.get("max_steps"),
            "socs": _soc_curve(rid),
        }
        if rid in REFERENCE_IDS:
            row["ref_label"] = REFERENCE_IDS[rid]
            row["status"] = _row_status(row, is_ref=True)
            ref.append(row)
        elif rc.get("type") in WM_TYPES:
            row["status"] = _row_status(row)
            wm.append(row)
    # Keep intentional analytic and unintended fallback rows in different groups.
    def key(r):
        backend = _rollout_backend(r)
        order = {
            "latent": 0,
            "analytic": 1,
            "fallback": 2,
            "model-free": 3,
        }.get(backend, 4)
        u = _metric_value(r, "utility")
        return (order, -(u if isinstance(u, (int, float)) else -1e9))

    wm.sort(key=key)
    return wm, ref


def _best(wm: list[dict]):
    best = {}
    for key, _, better, _ in METRICS:
        if better == "neutral":
            continue
        vals = [
            (r["id"], value)
            for r in wm
            if r.get("status") == "measured"
            and (value := _metric_value(r, key)) is not None
        ]
        if not vals:
            continue
        best[key] = (max if better == "higher" else min)(vals, key=lambda kv: kv[1])[0]
    return best


def _cell(r, key, fmt):
    value = _metric_value(r, key)
    return fmt.format(value) if value is not None else "—"


def render(wm: list[dict], ref: list[dict]) -> str:
    best = _best(wm)
    head_cells = "".join(f"<th>{lbl}</th>" for _, lbl, _, _ in METRICS)

    def row_html(r, is_ref=False):
        backend = _rollout_backend(r, is_ref=is_ref)
        sel_lbl, sel_cls = _selector(r["rc"], backend)
        duty = _jetson_duty(r["rc"])
        duty_s = f"1/{int(round(1/duty))}" if duty and duty < 1 else ("every step" if duty else "—")
        artifact = r["mean"].get("artifact_loaded")
        badge = ""
        if is_ref:
            badge = '<span class="ok">sub-watt OBC</span>'
        elif backend == "fallback":
            badge = '<span class="warn">fallback — excluded</span>'
        elif backend == "analytic":
            badge = '<span class="ok">intentional analytic</span>'
        elif r["rc"].get("type") == "lewm_cem_eventsat":
            badge = (
                '<span class="ok">artifact</span>' if artifact == 1.0 else '<span class="warn">fallback</span>'
            )
        elif r["rc"].get("type") == "dreamerv3_eventsat":
            badge = (
                '<span class="ok">policy</span>'
                if r["mean"].get("policy_loaded") == 1.0
                else '<span class="warn">untrained · heuristic stub</span>'
            )
        name = r.get("ref_label") if is_ref else sel_lbl
        cls = "ref-row" if is_ref else sel_cls
        cells = ""
        for key, _, _, fmt in METRICS:
            hit = (not is_ref) and best.get(key) == r["id"]
            cells += f'<td class="{"best" if hit else ""}">{_cell(r, key, fmt)}</td>'
        n = f'n={r["n"]}' if r["n"] else ""
        steps = r["steps"]
        dur = f'{steps//1440}d' if isinstance(steps, int) and steps % 1440 == 0 else (f"{steps} st" if steps else "")
        cfg = (
            f"{dur} · {n} · sub-watt rules on OBC"
            if is_ref
            else (
                f"{_config_summary(r['rc'], r.get('compute_w'), backend=backend)} "
                f"· planning {duty_s} · {dur} {n}"
            )
        )
        status = str(r.get("status", "notrun"))
        status_badge = (
            f"<span class=\"ok\">{status}</span>"
            if status == "measured"
            else f"<span class=\"warn\">{status}</span>"
        )
        backend_cls = backend.replace(" ", "-")
        return f"""<tr class="{cls}" data-backend="{backend_cls}">
      <td class="exp"><div class="sel">{name} {badge}</div>
        <div class="eid">{r['id']}</div>
        <div class="cfg">{cfg}</div></td>
      <td>{status_badge}</td>
      <td class="backend backend-{backend_cls}">{backend}</td>
      {cells}
      <td class="spark-cell">{_sparkline(r['socs'])}</td>
    </tr>"""

    wm_rows = "\n".join(row_html(r) for r in wm) or '<tr><td colspan="99" class="nodata">no WM runs yet — run a config under configs/experiments/world_model/</td></tr>'
    ref_rows = "\n".join(row_html(r, is_ref=True) for r in ref)
    ref_block = (
        f'<h2>Reference baseline</h2><table class="board"><thead><tr><th class="exp">Baseline</th><th>Status</th><th>Rollout backend</th>{head_cells}<th>SOC over run</th></tr></thead><tbody>{ref_rows}</tbody></table>'
        if ref
        else ""
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EventSat · World-Model Planner Board</title>
<style>
 :root {{ --line:#e3e6ea; --muted:#616b76; --blue:#0969da; --bg:#f6f8fa; }}
 * {{ box-sizing:border-box; }}
 body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   margin:0; color:#1f2328; background:var(--bg); }}
 header {{ padding:26px 40px 14px; }}
 h1 {{ margin:0 0 6px; font-size:22px; }}
 .sub {{ color:var(--muted); font-size:13.5px; max-width:1000px; line-height:1.5; }}
 main {{ padding:8px 40px 40px; }}
 h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:26px 0 8px; }}
 table.board {{ border-collapse:collapse; width:100%; background:#fff; border:1px solid var(--line);
   border-radius:12px; overflow:hidden; font-size:13px; }}
 .board th, .board td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
 .board th {{ background:#fbfcfd; color:var(--muted); font-weight:600; font-size:12px; text-align:right; }}
 .board th.exp, .board td.exp {{ text-align:left; white-space:normal; min-width:300px; }}
 .board tr:last-child td {{ border-bottom:none; }}
 td.best {{ background:#eaf7ee; font-weight:700; color:#0a5c2b; }}
 .sel {{ font-weight:650; font-size:13.5px; }}
 .eid {{ font-family:ui-monospace,monospace; font-size:11.5px; color:var(--blue); margin-top:1px; }}
 .cfg {{ color:var(--muted); font-size:11.5px; margin-top:3px; }}
 .sel-cem .sel {{ color:#0969da; }} .sel-analytic .sel {{ color:#1a7f37; }}
 .sel-fallback .sel {{ color:#9a6700; }} .sel-beam .sel {{ color:#8250df; }}
 .sel-rl .sel {{ color:#bc4c00; }}
 .board td.backend {{ text-align:left; font-weight:600; }}
 .backend-analytic {{ color:#1a7f37; }} .backend-fallback {{ color:#9a6700; }}
 .ref-row {{ background:#fbfbfc; color:var(--muted); }} .ref-row .sel {{ color:#57606a; }}
 .ok {{ font-size:10px; color:#0a5c2b; border:1px solid #accbb1; border-radius:4px; padding:0 5px; margin-left:4px; }}
 .warn {{ font-size:10px; color:#9a6200; border:1px solid #d9b271; border-radius:4px; padding:0 5px; margin-left:4px; }}
 .spark-cell {{ text-align:center; }}
 svg.spark .ref {{ stroke:#c9d1d9; stroke-width:1; stroke-dasharray:3 3; }}
 svg.spark .ref.safe {{ stroke:#f0c9c9; }}
 .nodata {{ color:#a0a8b0; font-size:12px; font-style:italic; }}
 .legend {{ margin-top:18px; color:var(--muted); font-size:12px; line-height:1.6; max-width:1000px; }}
 .legend b {{ color:#1f2328; }} code {{ font-family:ui-monospace,monospace; background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }}
 footer {{ color:var(--muted); font-size:11.5px; padding:0 40px 30px; }}
</style></head><body>
<header>
 <h1>EventSat · World-Model Planner Board</h1>
 <div class="sub">Onboard (AO) planners compared by the equations used for candidate rollouts:
 learned LeWM latent dynamics versus exact analytic EventSat transitions, using the same CEM search.
 Wall-clock planning latency and planner energy are measured by each run. The OBC- and Jetson-priced
 analytic variants are shown as separate cost assumptions; the board does not infer which is cheaper.
 Green = best among full-week, measured, non-fallback planner runs.</div>
</header>
<main>
 <h2>Latent and analytic MPC planners</h2>
 <table class="board"><thead><tr><th class="exp">Selector · run</th><th>Status</th><th>Rollout backend</th>{head_cells}<th>SOC over run</th></tr></thead>
 <tbody>{wm_rows}</tbody></table>
 {ref_block}
 <div class="legend">
  <b>Rollout backend</b> — <span style="color:#0969da">latent</span> uses the loaded LeWM artifact;
  <span style="color:#1a7f37">analytic</span> uses the canonical EventSat transitions and deterministic
  onboard contact/eclipse schedules. <span style="color:#9a6700">fallback</span> is an unintended
  artifact-load failure and is excluded from measured comparisons; it is never grouped with analytic.
  The symbolic controller is the sub-watt rule-based reference.<br>
  <b>Selector</b> — CEM (MPC): sample mode sequences and refine elites.
  <span style="color:#8250df">Beam (RL top-k)</span>: an RL/policy step proposes the top-k actions, each
  propagated through the LeWM predictor for H steps (same WM, different search). <span style="color:#bc4c00">DreamerV3</span>:
  model-free RL policy (heuristic fallback until a trained policy artifact is supplied).<br>
  <b>Planner cost</b> — ms/event is measured wall-clock time over actual planning events; Wh/episode
  accumulates the configured backend price only while planning. Analytic <code>obc</code> rows use the
  explicitly configured sub-watt price; <code>jetson</code> twins use +7&nbsp;W. These are measured
  simulator accounting fields, not an assumed performance conclusion.<br>
  <b>Planning duty</b> — fraction of steps that run CEM. <code>1/12</code> means plan once and execute
  the cached 12-step schedule before replanning.<br>
  <b>Run-id knobs</b> — <code>-h6</code> means <code>plan_hold=6</code> decision steps, so the planner replans
  every 6 minutes at the current 60&nbsp;s step size. <code>H</code> is the lookahead horizon, while
  <code>hold</code> is how many planned actions are committed before replanning; current sweeps use
  <code>H=max(12, hold)</code>. <code>-pc14</code> means the compute-power override is
  <code>Pplan=14 W</code> (<code>pc3p5</code> = 3.5&nbsp;W). <code>-mmdl</code>/<code>-mmsafe</code>
  are mission-mode retargeting presets for downlink/safe scoring. <code>256×4</code> means CEM samples ×
  CEM iterations.<br>
  <b>SOC curve</b> — battery state over the whole run; dashed lines mark reserve (0.5) and the safe floor (0.2).
 </div>
</main>
<footer>Auto-generated {ts} by <code>scripts/build_wm_board.py</code> (via <code>refresh_board.py</code>).
 Served on <code>:8801</code>. Add a run: drop a config in <code>configs/experiments/world_model/</code> and it appears here.</footer>
</body></html>"""


def main() -> None:
    wm, ref = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(wm, ref))
    print(f"wm board: {len(wm)} WM run(s), {len(ref)} reference → {OUT}")


if __name__ == "__main__":
    main()
