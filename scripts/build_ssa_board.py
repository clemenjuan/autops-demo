"""Build the SSA constellation organisation board: data/figures/ssa_board.html.

Replaces the retired flamingo-lite board (flamingo was dropped 2026-07-07;
SSA is the constellation scenario). Harvests every ``data/results/ssa_*``
run directly, groups by organisation x N x variant, and renders:

  1. The organisation x N sweep at nominal power (E-SSA1 baseline).
  2. The power-axis coordination-value sweep (E-SSA2, solar 12/16/20/24 W).
  3. Any RL-cell runs (tagged mock until PPO checkpoints exist).

    uv run python scripts/build_ssa_board.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path("data/results")
OUT = Path("data/figures/ssa_board.html")

RUN_RE = re.compile(
    r"^ssa_(?P<org>sas|cmas|dmas|hmas|imas)_(?P<paradigm>ao|ag|cg|ah)_"
    r"(?P<rep>symb|rl)_n(?P<n>\d+)(?:-sp(?P<sp>\d+))?$"
)
ORG_ORDER = ["sas", "cmas", "dmas", "hmas", "imas"]

METRIC_COLS = [
    ("ssa_delivered_coverage", "Delivered cov."),
    ("ssa_delivered_coverage_auc", "Coverage AUC"),
    ("duplicate_observation_rate", "Dup. rate"),
    ("observation_hours", "Obs. hours"),
    ("relayed_delivery_fraction", "Relay frac."),
    ("isl_connectivity", "ISL conn."),
    ("mean_delivery_latency_steps", "Deliv. lat. (steps)"),
    ("final_battery_soc", "Final SoC"),
    ("eta_scale", "M-10 η"),
]


def harvest() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("ssa_*/results.json")):
        run_id = path.parent.name
        match = RUN_RE.match(run_id)
        if not match:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stats = payload.get("experiment_statistics", {})
        mean = stats.get("mean", {}) if isinstance(stats, dict) else {}
        std = stats.get("std", {}) if isinstance(stats, dict) else {}
        config = payload.get("config", {})
        episodes = int(config.get("num_episodes", 0) or 0)
        steps = int(config.get("max_steps", 0) or 0)
        rep_cfg = config.get("representation_config", {}) or {}
        is_mock = bool(rep_cfg.get("rl_mock", False))
        rep = match.group("rep")
        if rep == "rl" and is_mock:
            status = "mock"
        elif episodes >= 30 and steps >= 10080:
            status = "measured"
        else:
            status = "smoke"
        rows.append({
            "run_id": run_id,
            "org": match.group("org"),
            "rep": rep,
            "n_sats": int(match.group("n")),
            "solar_w": int(match.group("sp")) if match.group("sp") else 24,
            "episodes": episodes,
            "steps": steps,
            "status": status,
            "mean": {k: mean.get(k) for k, _ in METRIC_COLS},
            "std": {k: std.get(k) for k, _ in METRIC_COLS},
        })
    return rows


def _fmt(value, key: str) -> str:
    if value is None:
        return "—"
    if key == "duplicate_observation_rate":
        return f"{value * 100:.1f}%"
    if key == "mean_delivery_latency_steps":
        return f"{value:.0f}"
    return f"{value:.3f}"


def _table(rows: list[dict], *, show_solar: bool) -> str:
    head = ["Org", "N"] + (["Solar W"] if show_solar else []) + \
        [label for _, label in METRIC_COLS] + ["n", "Status"]
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for row in rows:
        cells = [row["org"].upper(), str(row["n_sats"])]
        if show_solar:
            cells.append(str(row["solar_w"]))
        for key, _ in METRIC_COLS:
            value = _fmt(row["mean"].get(key), key)
            if key == "ssa_delivered_coverage" and row["std"].get(key) is not None:
                value += f" ± {row['std'][key]:.3f}"
            cells.append(value)
        cells.append(str(row["episodes"]))
        cells.append(f'<span class="st {row["status"]}">{row["status"]}</span>')
        num_start = 2 + (1 if show_solar else 0)
        tds = "".join(
            f'<td class="num">{c}</td>' if i >= num_start and not c.startswith("<span")
            else f"<td>{c}</td>"
            for i, c in enumerate(cells)
        )
        out.append(f"<tr>{tds}</tr>")
    out.append("</table>")
    return "".join(out)


def build(rows: list[dict]) -> str:
    def sort_key(row):
        return (ORG_ORDER.index(row["org"]), row["n_sats"], row["solar_w"])

    baseline = sorted(
        [r for r in rows if r["solar_w"] == 24 and r["rep"] == "symb"], key=sort_key
    )
    power = sorted([r for r in rows if r["rep"] == "symb"], key=sort_key)
    rl = sorted([r for r in rows if r["rep"] == "rl"], key=sort_key)

    measured = [r for r in rows if r["status"] == "measured"]
    best_cov = max(
        (r["mean"].get("ssa_delivered_coverage") or 0.0 for r in measured), default=0.0
    )
    kpis = (
        f'<div class="kpi"><b>{len(rows)}</b>runs harvested</div>'
        f'<div class="kpi"><b>{len(measured)}</b>measured (n≥30, full week)</div>'
        f'<div class="kpi"><b>{best_cov:.2f}</b>best delivered coverage</div>'
        f'<div class="kpi"><b>100</b>RSO catalog (near-orbit)</div>'
    )

    power_data = {}
    for row in power:
        if row["status"] != "measured":
            continue
        key = f'{row["org"].upper()} N={row["n_sats"]}'
        power_data.setdefault(key, []).append(
            [row["solar_w"], row["mean"].get("ssa_delivered_coverage")]
        )
    power_json = json.dumps(power_data)

    dup_data = json.dumps([
        [f'{r["org"].upper()} n{r["n_sats"]}',
         (r["mean"].get("duplicate_observation_rate") or 0.0) * 100.0]
        for r in baseline if r["status"] == "measured"
    ])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS — SSA constellation board</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<style>
 body {{ background:#fff; color:#111; margin:0; font-family:'Computer Modern Serif',Georgia,'Times New Roman',serif; }}
 header {{ padding:34px 56px 16px; border-bottom:2px solid #0065BD; }}
 h1 {{ margin:0; font-size:26px; font-weight:600; color:#0065BD; }}
 .sub {{ color:#444; margin-top:8px; font-size:13.5px; font-family:'Computer Modern Sans',Arial,sans-serif; max-width:1040px; line-height:1.5; }}
 section {{ padding:10px 56px 24px; max-width:1280px; }}
 h2 {{ font-size:17.5px; color:#005293; margin:26px 0 4px; font-weight:700; }}
 .caption {{ color:#555; font-size:13px; margin:2px 0 10px; }}
 table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
 th {{ text-align:left; padding:6px 10px; border-top:2px solid #111; border-bottom:1px solid #111; font-weight:600; }}
 td {{ padding:5px 10px; border-bottom:1px solid #e2e2e2; vertical-align:top; }}
 tr:last-child td {{ border-bottom:2px solid #111; }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .st {{ font-size:11px; padding:1px 7px; border:1px solid; border-radius:2px; white-space:nowrap; font-family:'Computer Modern Sans',Arial,sans-serif; }}
 .st.measured {{ color:#1e6b34; border-color:#1e6b34; }}
 .st.smoke {{ color:#9a6200; border-color:#9a6200; }}
 .st.mock {{ color:#888; border-color:#bbb; }}
 .kpis {{ display:flex; gap:26px; margin:10px 0 4px; flex-wrap:wrap; }}
 .kpi b {{ font-size:21px; display:block; color:#0065BD; }}
 .kpi {{ font-size:12px; color:#555; }}
 .guide {{ border-left:3px solid #0065BD; background:#f5f8fb; padding:10px 16px; font-size:13.5px; margin:8px 0; }}
 .plot {{ width:100%; height:360px; }}
 .twocol {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; }}
 @media (max-width:1100px){{ .twocol{{grid-template-columns:1fr;}} }}
</style></head><body>
<header>
 <h1>AUTOPS — SSA constellation organisation board</h1>
 <div class="sub">Organisation axis on the de-toyed SSA scenario (2026-07-07): shared-plane constellation,
 near-orbit M=100 RSO catalog, 52.7 km optical range with the Simera 32.2° FOV, sub-minute rate-limited ISL
 record relay, real Ottobrunn passes. Utility is delivered-to-ground RSO coverage (Collective-Negative).
 Replaces the retired flamingo-lite board. Rebuilt by <code>scripts/build_ssa_board.py</code> via
 <code>scripts/refresh_board.py</code>.</div>
</header>

<section>
 <div class="kpis">{kpis}</div>

 <h2>1&emsp;Organisation × N sweep — nominal power (E-SSA1)</h2>
 <div class="caption">Symbolic AO cells, n = 30 paired seeds, full week (10080 steps). Headline: organisation
 separates <i>efficiency</i> (duplicates, observing effort, relay use), not delivered coverage — independence
 buys the same utility at 2–3× the observing effort.</div>
 {_table(baseline, show_solar=False)}

 <h2>2&emsp;Coordination-value power axis (E-SSA2)</h2>
 <div class="caption">Solar generation swept 12/16/20 W against the 24 W baseline at N = 3. Coordination has an
 energy price (ISL relay TX): the org ranking inverts somewhere below 24 W — the crossover is the
 coordination-value boundary.</div>
 <div class="twocol">
  <div id="powerplot" class="plot"></div>
  <div id="dupplot" class="plot"></div>
 </div>
 {_table([r for r in power if r["solar_w"] != 24], show_solar=True)}

 <h2>3&emsp;RL cells</h2>
 <div class="caption">subsymbolic_ssa (32D obs, 8-mode action incl. isl_share). Mock until PPO checkpoints land
 (backlog item 17).</div>
 {_table(rl, show_solar=False)}
</section>

<script>
const powerData = {power_json};
const dupData = {dup_data};
const traces = Object.entries(powerData).map(([name, pts]) => ({{
  x: pts.map(p => p[0]), y: pts.map(p => p[1]),
  name, mode: 'lines+markers', type: 'scatter'
}}));
Plotly.newPlot('powerplot', traces, {{
  title: {{text: 'Delivered coverage vs solar generation', font: {{size: 14}}}},
  xaxis: {{title: 'Solar peak W', tickvals: [12, 16, 20, 24]}},
  yaxis: {{title: 'Delivered coverage', rangemode: 'tozero'}},
  margin: {{t: 40, r: 10}}, legend: {{orientation: 'h'}}
}}, {{displayModeBar: false}});
Plotly.newPlot('dupplot', [{{
  x: dupData.map(d => d[0]), y: dupData.map(d => d[1]), type: 'bar',
  marker: {{color: '#0065BD'}}
}}], {{
  title: {{text: 'Duplicate-observation rate at 24 W (%)', font: {{size: 14}}}},
  yaxis: {{title: 'Duplicates (%)'}}, margin: {{t: 40, r: 10}}
}}, {{displayModeBar: false}});
</script>
</body></html>
"""


def main() -> None:
    rows = harvest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(rows), encoding="utf-8")
    print(f"ssa board: {len(rows)} runs -> {OUT}")


if __name__ == "__main__":
    main()
