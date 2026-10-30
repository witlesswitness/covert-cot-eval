"""
figures.py — the study's figures, styled to match the previous study.

  fig1  leak rate by covertness level
  fig2  the reasoning-monitor advantage: fewer false positives + better at
        catching intent that leaves no action-level trace
  fig3  chain-of-thought faithfulness under concealment pressure, by model
  fig4  leak rate by agent model
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import load, wilson, THRESH

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

RISK, RISK_LT, GOOD, GOOD_LT, GRID = "#e76f51", "#f4a582", "#2a9d8f", "#83c5be", "#dddddd"
WILSON = "Error bars are 95% Wilson confidence intervals."

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 140,
})


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return p * 100, (p - lo) * 100, (hi - p) * 100


def flagged(e, view):
    s = e.get(view, {}).get("suspicion")
    return s is not None and s >= THRESH


def save(fig, name, caption=None):
    if caption:
        fig.text(0.5, -0.03, caption, ha="center", fontsize=8, color="#666666")
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


eps = load()
injected = [e for e in eps if e["level"] is not None]
clean = [e for e in eps if e["level"] is None]
leaked = [e for e in injected if e["leaked"]]

# --- Figure 1: leak rate by covertness ---
names = ["overt", "encoded", "disguised"]
rs, le, ue = [], [], []
for lv in (0, 1, 2):
    sub = [e for e in injected if e["level"] == lv]
    r, l, u = rate(sum(e["leaked"] for e in sub), len(sub))
    rs.append(r); le.append(l); ue.append(u)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(names, rs, color=[RISK_LT, RISK_LT, RISK], yerr=[le, ue], capsize=4, error_kw={"ecolor": "#555"})
ax.set_ylabel("Leak rate (%)")
ax.set_title("Leak rate by covertness level")
ax.set_ylim(0, 112)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for i, r in enumerate(rs):
    ax.text(i, r + ue[i] + 1.5, f"{r:.0f}%", ha="center", fontsize=10)
save(fig, "fig1_leak_by_covertness.png", caption=WILSON)

# --- Figure 2: the reasoning-monitor advantage ---
# FP on clean, and attempt-detection (intent present, no completed leak)
attempts = [e for e in injected if not e["leaked"] and e.get("judge", {}).get("reveals_intent")]
cats = ["False positives\n(clean episodes)", "Attempt detection\n(intent, no leak)"]
act = [rate(sum(flagged(e, "mon_actions") for e in clean), len(clean)),
       rate(sum(flagged(e, "mon_actions") for e in attempts), len(attempts))]
rea = [rate(sum(flagged(e, "mon_reasoning") for e in clean), len(clean)),
       rate(sum(flagged(e, "mon_reasoning") for e in attempts), len(attempts))]
x = np.arange(len(cats)); width = 0.36
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - width / 2, [a[0] for a in act], width, label="Actions only", color=RISK_LT,
       yerr=[[a[1] for a in act], [a[2] for a in act]], capsize=4, error_kw={"ecolor": "#555"})
ax.bar(x + width / 2, [r[0] for r in rea], width, label="With reasoning", color=GOOD,
       yerr=[[r[1] for r in rea], [r[2] for r in rea]], capsize=4, error_kw={"ecolor": "#555"})
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("% flagged at suspicion ≥ 50")
ax.set_title("What the reasoning view adds")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_ylim(0, 112)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for xi, a in zip(x - width / 2, act):
    ax.text(xi, a[0] + a[2] + 2, f"{a[0]:.0f}%", ha="center", fontsize=9)
for xi, r in zip(x + width / 2, rea):
    ax.text(xi, r[0] + r[2] + 2, f"{r[0]:.0f}%", ha="center", fontsize=9)
save(fig, "fig2_reasoning_advantage.png", caption=WILSON)

# --- Figure 3: CoT faithfulness under concealment, by model ---
models = sorted({e["model"] for e in leaked})
mlabels = [m.split("/")[-1] for m in models]
x = np.arange(len(models)); width = 0.36
fig, ax = plt.subplots(figsize=(7, 4.2))
for j, (cz, color, lab) in enumerate([(False, GOOD_LT, "concealment off"), (True, GOOD, "concealment on")]):
    rs, le, ue = [], [], []
    for m in models:
        sub = [e for e in leaked if e["model"] == m and e["conceal"] == cz
               and e.get("judge", {}).get("reveals_intent") is not None]
        r, l, u = rate(sum(e["judge"]["reveals_intent"] for e in sub), len(sub))
        rs.append(r); le.append(l); ue.append(u)
    xoff = x + (j - 0.5) * width
    ax.bar(xoff, rs, width, label=lab, color=color, yerr=[le, ue], capsize=4, error_kw={"ecolor": "#555"})
    for xi, r in zip(xoff, rs):
        ax.text(xi, r + 1.5, f"{r:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(mlabels)
ax.set_ylabel("Reasoning reveals intent (%)")
ax.set_title("Chain-of-thought faithfulness under concealment pressure")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_ylim(0, 112)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, "fig3_faithfulness.png", caption=WILSON + "  Among leaked episodes.")

# --- Figure 4: leak rate by agent model ---
rs, le, ue = [], [], []
for m in models:
    sub = [e for e in injected if e["model"] == m]
    r, l, u = rate(sum(e["leaked"] for e in sub), len(sub))
    rs.append(r); le.append(l); ue.append(u)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(mlabels, rs, color=RISK, yerr=[le, ue], capsize=4, error_kw={"ecolor": "#555"})
ax.set_ylabel("Leak rate (%), injected episodes")
ax.set_title("Leak rate by agent model")
ax.set_ylim(0, 112)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for i, r in enumerate(rs):
    ax.text(i, r + ue[i] + 1.5, f"{r:.0f}%", ha="center", fontsize=10)
save(fig, "fig4_leak_by_model.png", caption=WILSON)

print("\nAll figures written to figures/")
