"""Module-time charts for one bench run.

One chart per profiled cell, plus a table. A sector is the median of that
module's time across the successful repetitions, as a share of the sum of those
medians. The time shown for the cell is the median of the full runs. ``sr`` is
its own module. Module colors stay fixed across charts. Rows are
distilled checkpoints first, then the others, lite before pro, and SD
then Full HD within each checkpoint.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

_MODULES = (
    "dit",
    "vae_decoder",
    "text_encoder",
    "vocoder",
    "audio_vae_decoder",
    "text_encoder_2",
    "sr",
)
_COLORS = {
    "dit": "#2563EB",
    "vae_decoder": "#059669",
    "text_encoder": "#EA580C",
    "vocoder": "#DB2777",
    "audio_vae_decoder": "#7C3AED",
    "text_encoder_2": "#CA8A04",
    "sr": "#0891B2",
    "other": "#CBD5E1",
}
_EXTRA_COLORS = ("#0F766E", "#B45309", "#BE123C", "#4338CA", "#0369A1", "#4D7C0F")
_RESOLUTION_ORDER = {"sd": 0, "hd": 1, "fhd": 2}
_REPORT_NAME = "module-time-report.html"

_LIGHT = (
    "--ink: #0f172a; --muted: #64748b; --line: #e2e8f0; --paper: #f8fafc; "
    "--card: #ffffff; --lead: #334155; --head: #f8fafc;"
)
_DARK = (
    "--ink: #e2e8f0; --muted: #94a3b8; --line: #334155; --paper: #0f172a; "
    "--card: #111827; --lead: #cbd5e1; --head: #1e293b;"
)
_CSS = f"""
  :root {{ color-scheme: light; {_LIGHT} }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.45 "Segoe UI", system-ui, sans-serif; }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 64px; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: -0.03em; }}
  h2 {{ margin: 36px 0 12px; font-size: 20px; letter-spacing: -0.02em; }}
  .lead {{ margin: 0; max-width: 78ch; color: var(--lead); }}
  .lead + .lead {{ margin-top: 8px; }}
  code {{ font-size: 0.92em; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin: 18px 0 8px; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; }}
  .legend-item i {{ width: 12px; height: 12px; border-radius: 99px; display: inline-block; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 14px 8px; }}
  .card h3 {{ margin: 0; font-size: 16px; }}
  .meta {{ margin: 4px 0 0; min-height: 2.8em; color: var(--muted); font-size: 12px; }}
  .card svg {{ display: block; width: min(100%, 250px); margin: 0 auto; }}
  .slice {{ fill: #ffffff; }}
  .slice.other {{ fill: #1e293b; }}
  .hub {{ fill: var(--ink); }}
  .hub-sub {{ fill: var(--muted); }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: var(--card); }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  thead th {{ font-weight: 650; background: var(--head); }}
  thead th.group {{ text-align: center; }}
  thead th.corner {{ text-align: left; vertical-align: bottom; }}
  th.split, td.split {{ border-left: 1px solid var(--line); }}
  tbody th {{ text-align: left; font-weight: 600; background: var(--card); min-width: 168px; }}
  .run {{ display: block; }}
  .sub {{ display: block; margin-top: 2px; color: var(--muted); font-weight: 450; font-size: 11px; }}
  td {{ font-weight: 650; }}
  td.missing {{ color: var(--muted); font-weight: 450; }}
  .share, .scale span {{ background: var(--heat); color: var(--on); }}
  .scale {{ display: flex; gap: 4px; margin-top: 10px; max-width: 520px; }}
  .scale span {{ flex: 1; text-align: center; font-size: 11px; font-weight: 650; padding: 6px 0; border-radius: 6px; }}
  .note {{ margin: 10px 0 0; color: var(--muted); font-size: 12px; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ color-scheme: dark; {_DARK} }}
    body:not(.vscode-light) .share, body:not(.vscode-light) .scale span {{ background: var(--heat-dark); color: var(--on-dark); }}
  }}
  body.vscode-dark, body.vscode-high-contrast {{ color-scheme: dark; {_DARK} }}
  body.vscode-dark .share, body.vscode-dark .scale span,
  body.vscode-high-contrast .share, body.vscode-high-contrast .scale span {{ background: var(--heat-dark); color: var(--on-dark); }}
  body.vscode-light {{ color-scheme: light; {_LIGHT} }}
  body.vscode-light .share, body.vscode-light .scale span {{ background: var(--heat); color: var(--on); }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 20px 14px 40px; }} }}
  @media print {{
    :root, body {{ color-scheme: light; {_LIGHT} }}
    .share, .scale span {{ background: var(--heat) !important; color: var(--on) !important; }}
    .card, .table-wrap {{ break-inside: avoid; }}
    main {{ max-width: none; }}
  }}
"""


def write_module_time_report(run: Path) -> Path | None:
    """Write ``module-time-report.html`` into ``run``. Returns the path, or None when there are no profiles."""
    rows = _load_rows(run)
    if not rows:
        return None
    target = run / _REPORT_NAME
    target.write_text(_page(run.name, rows), encoding="utf-8")
    return target


def _row_order(name: str) -> tuple[int, int, int, str]:
    """Distilled before full, lite before pro, then SD, Full HD."""
    distilled = 0 if "-distill" in name else 1
    if name.startswith("lite"):
        family = 0
    elif name.startswith("pro"):
        family = 1
    else:
        family = 2
    resolution = name.rsplit("-", 1)[-1]
    return (distilled, family, _RESOLUTION_ORDER.get(resolution, 9), name)


def _load_rows(run: Path) -> list[dict]:
    directory = run / "profiles"
    if not directory.is_dir():
        return []
    selected = _cells_from_results(run)
    if selected:
        cells = selected
    else:
        latest: dict[str, Path] = {}
        for path in sorted(directory.glob("*.json")):
            job, sep, _rest = path.stem.partition("_")
            latest[job if sep else path.stem] = path
        cells = [(name, [path], None) for name, path in latest.items()]
    rows = []
    for name, paths, seconds in cells:
        payloads = [_read_profile(path) for path in paths]
        payloads = [payload for payload in payloads if payload is not None]
        row = _combine(name, payloads, seconds)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: _row_order(item["name"]))
    return rows


def _cells_from_results(run: Path) -> list[tuple[str, list[Path], float | None]]:
    """Successful cells, with every repetition that entered the median. Failures are left out."""
    path = run / "results.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    cells: list[tuple[str, list[Path], float | None]] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("success"):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        raw_paths = item.get("profiles")
        if not isinstance(raw_paths, list) or not raw_paths:
            single = item.get("profile")
            raw_paths = [single] if isinstance(single, str) else []
        paths = [
            Path(item_path)
            for item_path in raw_paths
            if isinstance(item_path, str) and Path(item_path).is_file()
        ]
        if not paths:
            continue
        seconds = item.get("seconds")
        cells.append((name, paths, float(seconds) if isinstance(seconds, (int, float)) else None))
    return cells


def _read_profile(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "time_mem_module":
        return None
    return payload


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _call_times(payload: dict) -> dict[str, float] | None:
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        return None
    modules = measurements.get("modules")
    if not isinstance(modules, dict):
        return None
    total = measurements.get("total_time")
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    times = {
        module: float(entry["time"])
        for module, entry in modules.items()
        if isinstance(entry, dict) and isinstance(entry.get("time"), (int, float))
    }
    profile = float(total)
    sr_raw = measurements.get("sr_time")
    sr = float(sr_raw) if isinstance(sr_raw, (int, float)) and sr_raw > 0 else 0.0
    other = max(0.0, profile - sum(times.values()))
    times["sr"] = sr
    times["other"] = other
    return times


def _combine(name: str, payloads: list[dict], seconds: float | None) -> dict | None:
    """Median of each module across calls. Shares are of the sum of those medians."""
    paired = []
    for payload in payloads:
        call = _call_times(payload)
        if call is not None:
            paired.append((payload, call))
    if not paired:
        return None
    calls = [call for _payload, call in paired]
    keys = {key for call in calls for key in call}
    times = {key: _median([call.get(key, 0.0) for call in calls]) for key in keys}
    span = sum(times.values())
    shares = {key: (value / span * 100.0 if span else 0.0) for key, value in times.items()}
    payload = paired[-1][0]
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    memory = {
        "generate": _max_peak(_peak_pair(item, "total_peak_allocated_mem", "total_peak_reserved_mem") for item, _call in paired),
        "sr": _max_peak(_peak_pair(item, "sr_peak_allocated_mem", "sr_peak_reserved_mem") for item, _call in paired),
    }
    gpu_bytes = _max_peak(
        (
            (0, int(env["gpu_total_mem"]))
            for item, _call in paired
            if isinstance((env := item.get("environment")), dict) and isinstance(env.get("gpu_total_mem"), int)
        )
    )
    if seconds is None:
        totals = []
        for call_payload, call in paired:
            measured = call_payload.get("measurements")
            total = measured.get("total_time") if isinstance(measured, dict) else None
            if isinstance(total, (int, float)):
                totals.append(float(total) + call.get("sr", 0.0))
        seconds = _median(totals) if totals else span
    return {
        "name": name,
        "total": seconds,
        "times": times,
        "shares": shares,
        "width": run.get("width"),
        "height": run.get("height"),
        "frames": run.get("latent_frames"),
        "steps": run.get("num_steps"),
        "guidance": run.get("guidance_weight"),
        "cache": run.get("cache_mode"),
        "gpu": environment.get("gpu"),
        "offload": run.get("offload_strategy") or "",
        "memory": memory,
        "gpu_bytes": 0 if gpu_bytes is None else gpu_bytes[1],
    }


def _columns(rows: list[dict]) -> list[str]:
    present = {module for row in rows for module in row["times"] if module != "other"}
    extras = sorted(module for module in present if module not in _MODULES)
    return [module for module in _MODULES if module in present] + extras + ["other"]


def _colors(columns: list[str]) -> dict[str, str]:
    colors = dict(_COLORS)
    extras = [name for name in columns if name not in colors]
    for index, name in enumerate(extras):
        colors[name] = _EXTRA_COLORS[index % len(_EXTRA_COLORS)]
    return colors


def _pct(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.01:
        return "<0.01"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _annular(cx: float, cy: float, inner: float, outer: float, start: float, end: float) -> str:
    sweep = end - start
    if sweep <= 0:
        return ""
    if sweep >= 359.999:
        end = start + 359.999
        sweep = 359.999
    large = 1 if sweep > 180 else 0

    def point(radius: float, angle: float) -> tuple[float, float]:
        radians = math.radians(angle)
        return cx + radius * math.sin(radians), cy - radius * math.cos(radians)

    x0, y0 = point(outer, start)
    x1, y1 = point(outer, end)
    xi1, yi1 = point(inner, end)
    xi0, yi0 = point(inner, start)
    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {outer:.3f} {outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} "
        f"L {xi1:.3f} {yi1:.3f} "
        f"A {inner:.3f} {inner:.3f} 0 {large} 0 {xi0:.3f} {yi0:.3f} Z"
    )


def _chart(row: dict, columns: list[str], colors: dict[str, str]) -> str:
    size = 232
    cx = cy = size / 2
    outer, inner = 96, 60
    angle = 0.0
    parts: list[str] = []
    for key in columns:
        share = row["shares"].get(key, 0.0)
        sweep = share / 100.0 * 360.0
        if sweep > 0.05:
            path = _annular(cx, cy, inner, outer, angle, angle + sweep)
            tip = f"{key}: {row['times'].get(key, 0.0):.2f} s, {share:.2f}%"
            parts.append(f'<path d="{path}" fill="{colors[key]}"><title>{html.escape(tip)}</title></path>')
            if share >= 8:
                mid = math.radians(angle + sweep / 2)
                lx = cx + ((inner + outer) / 2) * math.sin(mid)
                ly = cy - ((inner + outer) / 2) * math.cos(mid)
                slice_class = "slice other" if key == "other" else "slice"
                parts.append(
                    f'<text class="{slice_class}" x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
                    f'dominant-baseline="central" font-size="13" font-weight="650">{_pct(share)}%</text>'
                )
        angle += sweep
    parts.append(
        f'<text class="hub" x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'font-size="18" font-weight="680">{row["total"]:.1f} s</text>'
        f'<text class="hub-sub" x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="11">median</text>'
    )
    label = html.escape(row["name"])
    return f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{label}">{"".join(parts)}</svg>'


def _meta(row: dict) -> str:
    bits = []
    if row["width"] is not None and row["height"] is not None:
        bits.append(f"{row['width']}×{row['height']}")
    if row["frames"] is not None:
        bits.append(f"{row['frames']} latent")
    if row["steps"] is not None:
        bits.append(f"{row['steps']} steps")
    if isinstance(row["guidance"], (int, float)):
        bits.append(f"cfg {row['guidance']:g}")
    if row["cache"]:
        bits.append(str(row["cache"]))
    return " · ".join(bits)


_HEAT_LIGHT = (
    (0.00, (247, 248, 250)),
    (0.08, (219, 234, 254)),
    (0.20, (147, 197, 253)),
    (0.40, (59, 130, 246)),
    (0.70, (29, 78, 216)),
    (1.00, (30, 58, 138)),
)
_HEAT_DARK = (
    (0.00, (15, 23, 42)),
    (0.08, (30, 41, 59)),
    (0.20, (30, 58, 138)),
    (0.40, (29, 78, 216)),
    (0.70, (37, 99, 235)),
    (1.00, (59, 130, 246)),
)


def _blend(stops: tuple[tuple[float, tuple[int, int, int]], ...], t: float) -> tuple[int, int, int]:
    rgb = stops[-1][1]
    for (left, start), (right, end) in zip(stops, stops[1:], strict=False):
        if t <= right:
            blend = 0.0 if right == left else (t - left) / (right - left)
            return tuple(
                round(start[channel] + (end[channel] - start[channel]) * blend) for channel in range(3)
            )
    return rgb


def _rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]}, {color[1]}, {color[2]})"


def _heat_style(share: float) -> str:
    """Light and dark cell colors. The stylesheet picks one from the theme."""
    t = max(0.0, min(1.0, share / 100.0))
    light = _rgb(_blend(_HEAT_LIGHT, t))
    dark = _blend(_HEAT_DARK, t)
    light_ink = "#ffffff" if t >= 0.34 else "#0f172a"
    luminance = 0.2126 * dark[0] + 0.7152 * dark[1] + 0.0722 * dark[2]
    dark_ink = "#0f172a" if luminance > 160 else "#e2e8f0"
    return f"--heat:{light};--on:{light_ink};--heat-dark:{_rgb(dark)};--on-dark:{dark_ink}"


def _legend(columns: list[str], colors: dict[str, str]) -> str:
    return "".join(
        f'<span class="legend-item"><i style="background:{colors[name]}"></i>{html.escape(name)}</span>'
        for name in columns
    )


def _cards(rows: list[dict], columns: list[str], colors: dict[str, str]) -> str:
    cards = []
    for row in rows:
        cards.append(
            "<article class=\"card\">"
            f"<h3>{html.escape(row['name'])}</h3>"
            f"<p class=\"meta\">{html.escape(_meta(row))}</p>"
            f"{_chart(row, columns, colors)}"
            "</article>"
        )
    return "".join(cards)


def _table(rows: list[dict], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    body = []
    for row in rows:
        cells = []
        for name in columns:
            share = row["shares"].get(name, 0.0)
            seconds = row["times"].get(name, 0.0)
            cells.append(
                f'<td class="share" style="{_heat_style(share)}" title="{seconds:.2f} s">{_pct(share)}</td>'
            )
        detail = f"{row['total']:.1f} s"
        body.append(
            "<tr>"
            f'<th scope="row"><span class="run">{html.escape(row["name"])}</span>'
            f'<span class="sub">{html.escape(detail)}</span></th>'
            + "".join(cells)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr><th>run</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _peak_pair(payload: dict, allocated_key: str, reserved_key: str) -> tuple[int, int] | None:
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        return None
    allocated = measurements.get(allocated_key)
    reserved = measurements.get(reserved_key)
    if not isinstance(allocated, (int, float)) or not isinstance(reserved, (int, float)):
        return None
    return int(allocated), int(reserved)


def _max_peak(pairs) -> tuple[int, int] | None:
    present = [pair for pair in pairs if pair is not None]
    if not present:
        return None
    return max(pair[0] for pair in present), max(pair[1] for pair in present)


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f}"


def _memory_table(rows: list[dict]) -> str:
    """Maximum allocated and reserved high-water for the generation and for SR."""
    gpu_bytes = max((row.get("gpu_bytes") or 0 for row in rows), default=0)
    peaks = [
        value
        for row in rows
        for pair in (row.get("memory") or {}).values()
        if pair is not None
        for value in pair
    ]
    scale = gpu_bytes or (max(peaks) if peaks else 0)

    def value_cell(value: int | None, *, split: bool = False) -> str:
        classes = ["split"] if split else []
        if value is None:
            classes.append("missing")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            return f"<td{class_attr}>—</td>"
        share = value / scale * 100.0 if scale else 0.0
        classes.append("share")
        tip = f"{value / 1024**3:.2f} GiB"
        return (
            f'<td class="{" ".join(classes)}" style="{_heat_style(share)}" '
            f'title="{html.escape(tip)}">{_gib(value)}</td>'
        )

    def pair_cells(pair: tuple[int, int] | None, *, split: bool = False) -> str:
        if pair is None:
            return value_cell(None, split=split) + value_cell(None)
        return value_cell(pair[0], split=split) + value_cell(pair[1])

    body = []
    for row in rows:
        memory = row.get("memory") or {}
        offload = str(row.get("offload") or "")
        sub = f'<span class="sub">{html.escape(offload)}</span>' if offload else ""
        body.append(
            "<tr>"
            f'<th scope="row"><span class="run">{html.escape(row["name"])}</span>{sub}</th>'
            f'{pair_cells(memory.get("generate"))}{pair_cells(memory.get("sr"), split=True)}'
            "</tr>"
        )
    if gpu_bytes:
        color = f"that value as a share of this GPU ({gpu_bytes / 1024**3:.0f} GiB)"
    else:
        color = "that value as a share of the largest number in this table"
    lead = (
        "Each number is the maximum high-water across repetitions, in GiB. "
        f"The color is {color}. "
        "<code>generate</code> is the profiled generation, including the gaps between modules. "
        "<code>sr</code> is the whole super-resolution stage, including weight load and kernel warmup. "
        "A dash means that peak was not recorded."
    )
    header = (
        "<thead>"
        '<tr><th rowspan="2" class="corner">run</th>'
        '<th colspan="2" class="group">generate</th>'
        '<th colspan="2" class="group split">sr</th></tr>'
        "<tr><th>allocated</th><th>reserved</th>"
        '<th class="split">allocated</th><th>reserved</th></tr>'
        "</thead>"
    )
    return (
        "<h2>Peak memory, GiB</h2>"
        f'<p class="lead">{lead}</p>'
        '<div class="table-wrap"><table>'
        f"{header}<tbody>{''.join(body)}</tbody></table></div>"
    )


def _scale() -> str:
    parts = [f'<span style="{_heat_style(float(value))}">{value}%</span>' for value in (0, 10, 25, 50, 75, 100)]
    return f'<div class="scale">{"".join(parts)}</div>'


def _page(run_name: str, rows: list[dict]) -> str:
    columns = _columns(rows)
    colors = _colors(columns)
    gpu = next((row["gpu"] for row in rows if row["gpu"]), None)
    intro = f"Run <code>{html.escape(run_name)}</code>"
    if gpu:
        intro += f" on {html.escape(str(gpu))}"
    intro += "."
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Module time share — {html.escape(run_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>Module time share</h1>
  <p class="lead">{intro}</p>
  <p class="lead">Each sector is the median time of that module across the successful repetitions, as a share of the sum of those medians. The number in the center is the median of the full runs, and it is not that sum. The same module keeps the same color on every chart. <code>sr</code> is the super-resolution pass. <code>other</code> is profile time outside the measured modules. Sectors under 0.5% are nearly invisible on the ring.</p>
  <div class="legend">{_legend(columns, colors)}</div>
  <h2>Charts</h2>
  <div class="grid">{_cards(rows, columns, colors)}</div>
  <h2>Share of median module time, %</h2>
  {_table(rows, columns)}
  {_scale()}
  <p class="note">Cell color is the size of the percentage. Hover a cell or a sector to see the median seconds. The line under a run name is the median of the full runs.</p>
  {_memory_table(rows)}
</main>
</body>
</html>
"""
