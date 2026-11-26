"""Speed along the route for a rule-driver drive and a learned-speed drive of the same destinations (PNG).

  .venv/Scripts/python.exe rl/plot_town_pair.py <rule town or owner folder> <learned town or owner folder> <out.png>

One panel per destination leg: speed (m/s) against distance driven since the leg was set, both cars, with the posted
limit under the learned car as a grey step. Simulation ground truth, scoring only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DT = .05
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LEARNED, RULE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (dataviz reference palette)


def town_of(folder: Path) -> Path:
    if (folder / "result.json").exists():
        return folder
    return Path(json.loads((folder / "status.json").read_text())["town_output"])


def drive(town: Path):
    network = json.loads((town / "network.json").read_text())
    streets = {s["id"]: s for s in network["streets"]}
    connectors = {c["id"]: c for c in network["connectors"]}

    def limit(rid):
        if rid in streets:
            return streets[rid]["cruise"]
        c = connectors.get(rid)
        return None if c is None else min(streets[c["incoming"]]["cruise"], streets[c["outgoing"]]["cruise"])
    t, v, lim = [], [], []
    for line in (town / "actor_trace.jsonl").open(encoding="utf-8"):
        i = line.find('"actors": [')
        j = line.find('}, {', i)
        a = json.loads(line[i + 11:j + 1])
        k = line.find('"sim_s": ')
        t.append(float(line[k + 9:line.find(',', k)]))
        v.append(a["speed"])
        value = limit(a.get("road_id"))
        lim.append(np.nan if value is None else value)
    result = json.loads((town / "result.json").read_text())
    legs = result["camera_driving"]["navigation"]["destinations"]
    return np.array(t), np.array(v), np.array(lim), legs


def main(rule_dir, learned_dir, out):
    rule, learned = drive(town_of(Path(rule_dir))), drive(town_of(Path(learned_dir)))
    legs = [leg for leg in rule[3] if leg["arrived_s"] is not None]
    n = len(legs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n + .6), facecolor=SURFACE, squeeze=False)
    for ax, leg in zip(axes[:, 0], legs):
        ax.set_facecolor(SURFACE)
        top = 0.
        for (t, v, lim, all_legs), colour, name in ((rule, RULE, "rule driver"), (learned, LEARNED, "learned speed")):
            mine = next((l for l in all_legs if l["name"] == leg["name"]), None)
            if mine is None:
                continue
            end = mine["arrived_s"] if mine["arrived_s"] is not None else t[-1]
            m = (t >= mine["set_s"]) & (t <= end)
            d = np.cumsum(v[m]) * DT
            if name == "learned speed":
                ax.step(d, lim[m], where="post", color=INK2, lw=1.2, ls=(0, (4, 3)), label="posted limit")
            ax.plot(d, v[m], color=colour, lw=2, label=f"{name}: {end - mine['set_s']:.0f} s" +
                    ("" if mine["arrived_s"] is not None else " (not reached)"), solid_capstyle="round")
            top = max(top, np.nanmax(v[m]), np.nanmax(lim[m]))
        ax.set_ylim(0, top + .8)
        ax.set_title(f"To {leg['name']}", loc="left", color=INK, fontsize=11)
        ax.set_ylabel("speed (m/s)", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(loc="lower right", bbox_to_anchor=(1., 1.), ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    axes[-1, 0].set_xlabel("distance driven since the destination was set (m)", color=INK2, fontsize=9)
    fig.suptitle("Speed along the route: rule driver and learned speed, same destinations and seed (CARLA, simulation)",
                 x=.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
