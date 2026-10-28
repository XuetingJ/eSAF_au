"""Reproduce manuscript Figures 1, 3, 4, 5 and Supplementary Figure S3 from model outputs."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import esaf_model as m

FIG = m.ROOT / "figures"; FIG.mkdir(exist_ok=True)
rows = m.run(m.HEADLINE)
central = {y: m.state(y, *m.CENTRAL) for y in m.YEARS}
c2050 = np.array([r["net_cost"] for r in rows if r["year"] == 2050])
blue = "#1f6e94"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ---------------- Figure 3
plt.rcParams.update({"font.size": 12})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ys = m.YEARS; vals = [central[y]["net_cost"] for y in ys]
a.plot(ys, vals, color=blue, lw=2)
for y in (2025, 2030, 2040, 2050):
    a.plot(y, central[y]["net_cost"], "o", color=blue)
    a.annotate(f"A${central[y]['net_cost']:.2f}", (y, central[y]["net_cost"]), textcoords="offset points", xytext=(0, 12), ha="center")
a.set_ylim(6, 13.2); a.set_xlim(2023.3, 2052); a.set_xlabel("Year"); a.set_ylabel("Plant-gate cost (A$2026 kg⁻¹)")
a.set_title("A  Central cost trajectory", loc="left", fontweight="bold")
p10, med, p90 = np.percentile(c2050, [10, 50, 90]); lo, hi = c2050.min(), c2050.max(); cen = central[2050]["net_cost"]
b.plot([lo, hi], [0, 0], color=blue, lw=1.2); b.plot([p10, p90], [0, 0], color=blue, lw=8, alpha=.55, solid_capstyle="butt")
b.plot(med, 0, "D", color=blue, ms=10); b.plot(cen, 0, "o", mfc="white", mec="black", ms=10)
for x, lab, dy in [(lo, "Min", -1), (hi, "Max", -1), (p10, "P10", 1), (p90, "P90", 1), (med, "Median", -1), (cen, "Central", 2)]:
    b.text(x, 0.55 * dy if dy != 2 else 1.25, f"{lab}\n{x:.2f}", ha="center", va="center", fontsize=12)
b.text(7.5, -1.6, "81 equally weighted cases\nDescriptive percentiles, not probability intervals", ha="center", fontsize=12)
b.set_ylim(-2.5, 2); b.set_yticks([]); b.spines[["left", "top", "right"]].set_visible(False); b.set_xlim(4.4, 11.3)
b.set_xlabel("2050 plant-gate cost (A$2026 kg⁻¹)"); b.set_title("B  Structured 2050 cost spread", loc="left", fontweight="bold")
a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "Figure3_cost_trajectory_and_2050_spread.png", dpi=300); plt.close(fig)

# ---------------- Figure 4
plt.rcParams.update({"font.size": 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(10.24, 4.975))
cols = {"capital": "#1f77b4", "electricity": "#ff7f0e", "co2": "#2ca02c", "fixed_om": "#d62728"}
names = {"capital": "Annualised capital", "electricity": "Electricity", "co2": "CO₂ supply", "fixed_om": "Fixed O&M"}
for i, y in enumerate((2025, 2030)):
    base = 0
    for k in cols:
        v = central[y][k]; a.bar(i, v, bottom=base, color=cols[k], width=.62, label=names[k] if i == 0 else None); base += v
    a.bar(i, central[y]["coproduct_credit"], color="#9467bd", width=.62, label="Co-product credit" if i == 0 else None)
    a.plot(i, central[y]["net_cost"], "D", color="#8c564b", ms=9, label="Net plant-gate cost" if i == 0 else None)
    a.annotate(f"A${central[y]['net_cost']:.2f}", (i, central[y]["net_cost"]), xytext=(0, 14), textcoords="offset points",
               ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#f6dede", ec="none"))
a.axhline(0, lw=1); a.set_xticks([0, 1], ["2025", "2030"]); a.set_ylabel("Cost contribution (A$2026 kg⁻¹ eSAF)")
a.set_title("A   Central cost decomposition", loc="left", fontweight="bold"); a.grid(axis="y", alpha=.3)
a.spines[["top", "right"]].set_visible(False)
a.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=2, frameon=False, fontsize=8)
drops = [(names[k], central[2025][k] - central[2030][k]) for k in ("electricity", "capital", "fixed_om", "co2")]
yy = np.arange(len(drops))[::-1]
b.barh(yy, [d for _, d in drops], color="#1f77b4"); b.set_yticks(yy, [n for n, _ in drops])
for y_, (_, d) in zip(yy, drops): b.text(d + .03, y_, f"{d:.2f}", va="center")
tot = central[2025]["net_cost"] - central[2030]["net_cost"]
b.text(.98, .1, f"Total decline: A${tot:.2f} kg⁻¹\n\nCo-product credit unchanged", transform=b.transAxes, ha="right", fontsize=8.5)
b.set_xlabel("Contribution to 2025–2030 cost decline\n(A$2026 kg⁻¹ eSAF)"); b.grid(axis="x", alpha=.3)
b.set_title("B   What drives the early decline?", loc="left", fontweight="bold"); b.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "Figure4_cost_decomposition.png", dpi=200); plt.close(fig)

# ---------------- Figure 5
shares = {y: m.factorial_shares({(r["electricity_path"], r["wacc"], r["ael_learning_rate"], r["ft_learning_rate"]): r["net_cost"]
                                  for r in rows if r["year"] == y}) for y in m.YEARS}
series = {"Electricity path": [shares[y]["electricity"] for y in ys], "WACC": [shares[y]["wacc"] for y in ys],
          "AEL learning": [shares[y]["ael"] for y in ys], "FT learning": [shares[y]["ft"] for y in ys],
          "Global–local interactions": [shares[y]["GROUP global-local interactions"] for y in ys]}
series["Other interactions / higher order"] = [max(0.0, 100 - sum(s[i] for s in series.values())) for i in range(len(ys))]
fig, ax = plt.subplots(figsize=(9.883, 5.883))
ax.stackplot(ys, *series.values(), labels=series.keys(), colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
ax.set_xlim(2025, 2050); ax.set_ylim(0, 100); ax.set_xlabel("Year"); ax.set_ylabel("Share of structured factorial sum of squares (%)")
ax.set_title("Annual sources of structured eSAF cost variation"); ax.legend(loc="center left", bbox_to_anchor=(1.01, .5))
fig.tight_layout(); fig.savefig(FIG / "Figure5_factorial_shares.png", dpi=300); plt.close(fig)

# ---------------- Figure S3
fig, ax = plt.subplots(figsize=(7.9, 5.383))
for w in m.FACTORS[1]:
    mm = [np.mean([r["net_cost"] for r in rows if r["year"] == 2050 and r["wacc"] == w and r["ael_learning_rate"] == lr]) for lr in m.FACTORS[2]]
    ax.plot([100 * x for x in m.FACTORS[2]], mm, label=f"{w*100:.0f}% WACC")
ax.set_xlabel("AEL learning rate (%)"); ax.set_ylabel("Mean 2050 plant-gate cost (AUD2026/kg eSAF)")
ax.set_title("WACC changes the unit-cost value of AEL learning"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "FigureS3_wacc_ael_interaction.png", dpi=300); plt.close(fig)

# ---------------- Figure 1 (process diagram; procured-CO2 boundary)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "mathtext.fontset": "dejavuserif"})
ink, dblue, green, gray, pale = "#242424", "#345d78", "#4f6958", "#6b6b6b", "#f1f4f6"
fig, ax = plt.subplots(figsize=(6.2, 4.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
fig.subplots_adjust(left=.015, right=.985, bottom=.01, top=.99)
def box(x, y, w, h, t, fs=9, fc="white", ec=ink, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=.9, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=ink, linespacing=1.3)
def arrow(p, q, c=ink):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1, color=c, shrinkA=0, shrinkB=0))
gross = m.P["load_electrolysis"] + m.auxiliary_mw(m.HEADLINE)
ax.text(5, 7.73, "Reference AEL–RWGS–FT configuration", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(5, 7.25, "2.50 t eSAF h⁻¹  ×  8,088 h y⁻¹  =  20.22 kt eSAF y⁻¹", ha="center", fontsize=9)
box(.22, 5.78, 2.60, .85, "Purchased electricity\nGross cost basis", 8.7, fc=pale, ec=dblue)
ax.plot([2.82, 9.45], [6.14, 6.14], color=dblue, lw=1); arrow((2.3, 5.78), (2.3, 4.88), dblue)
for x in [4.15, 6.35, 8.55]: arrow((x, 6.14), (x, 4.88), dblue)
ax.text(6.3, 6.45, f"Plant electricity demand: {gross:.2f} MW in 2025", ha="center", fontsize=8.7, color=dblue)
box(.18, 3.76, 1.08, .75, "Water", 8.8); box(1.52, 3.57, 1.55, 1.3, "Alkaline\nelectrolysis\n$K_{EL}$", 9, ec=dblue)
arrow((1.26, 4.13), (1.52, 4.13)); box(3.48, 3.57, 1.45, 1.3, "RWGS", 9.4)
arrow((3.07, 4.23), (3.48, 4.23)); ax.text(3.27, 4.5, "H₂", ha="center", fontsize=8.3)
box(3.12, 1.95, 2.18, .88, "Procured CO₂\ncompressed, at fence", 7.6, ec=green); arrow((4.2, 2.80), (4.2, 3.57), green)
box(5.67, 3.57, 1.45, 1.3, "FT\nsynthesis", 9); arrow((4.93, 4.23), (5.67, 4.23))
ax.text(5.3, 4.95, "Syngas\nCO / H₂", ha="center", va="bottom", fontsize=7.7)
box(7.8, 3.57, 1.83, 1.3, "Upgrading and\nfractionation", 8.8); arrow((7.12, 4.23), (7.8, 4.23))
ax.text(7.46, 4.95, "Syncrude", ha="center", va="bottom", fontsize=7.7)
ax.plot([3.37, 3.37, 9.72, 9.72], [3.45, 3.13, 3.13, 3.45], color=gray, lw=.9)
ax.text(6.6, 2.91, "Installed synthesis and upgrading  $K_{SYN}$", ha="center", fontsize=8.3)
ax.plot([9.63, 9.86, 9.86], [4.1, 4.1, 2.54], color=ink, lw=1); arrow((9.86, 2.54), (9.86, 2.47))
box(7.45, 1.57, 2.5, .9, "Neat eSAF\nDiesel and naphtha", 8.7, fc=pale)
box(.22, 1.5, 2.8, 1.13, "PEM fuel-cell recovery\n1.95 MW\nNo headline electricity credit", 7.8, ec=dblue, ls="--")
ax.text(5.6, 1.45, "Conceptual process flows;\nsource engineering data [8].", ha="center", fontsize=8.0, color=gray)
box(.22, .25, 9.73, .72, "Residual site integration and balance of plant  $K_{BOP}$", 8.9, fc="#f7f7f7", ec=gray)
ax.text(5, .02, "CO₂ capture and compression lie with the supplier; BOP is intended for costs outside the installed packages.",
        ha="center", va="bottom", fontsize=7.4, color=gray)
fig.savefig(FIG / "Figure1_process_configuration.png", dpi=350, bbox_inches="tight", pad_inches=.06); plt.close(fig)
print("figures written to", FIG)
