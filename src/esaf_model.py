"""Dynamic plant-gate techno-economic model of Australian eSAF production, 2025-2050.

Implements Eqs. (1)-(9) of the manuscript. All inputs are read from data/.
Monetary values are real 2026 Australian dollars (A$).
"""
from __future__ import annotations
import csv, itertools, math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data"
YEARS = list(range(2025, 2051))


def _read(name):
    with open(INPUTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


P = {r["parameter"]: float(r["value"]) for r in _read("parameters.csv")}
EXP = {int(r["year"]): r for r in _read("global_experience.csv")}
ELEC = {int(r["year"]): r for r in _read("electricity_prices.csv")}
DOM = {int(r["year"]): r for r in _read("domestic_experience.csv")}
LEVELS = {r["factor"]: r["levels"].split(";") for r in _read("factor_levels.csv")}
FACTORS = [LEVELS["electricity_path"],
           [float(x) for x in LEVELS["wacc"]],
           [float(x) for x in LEVELS["ael_learning_rate"]],
           [float(x) for x in LEVELS["ft_learning_rate"]]]
CENTRAL = ("Central", 0.07, 0.132, 0.05)


@dataclass(frozen=True)
class Boundary:
    """Accounting choices. Headline: procured CO2 (capture and compression outside the plant)."""
    procured_co2: bool = True        # screening assumption; False = legacy accounting (PSA capture + CO2 compression loads retained; not a costed on-site capture design)
    credit_pem: bool = False         # True = net 1.95 MW PEM fuel-cell recovery against purchases
    include_bop: bool = True         # False = remove the US$80 million BOP allowance
    legacy_rounded_residual: bool = False  # True = use 201.56 - 188.28 MW (legacy archive arithmetic)


HEADLINE = Boundary()
LEGACY = Boundary(procured_co2=False, legacy_rounded_residual=True)

Q = P["esaf_output_rate"] * 1000 * P["operating_hours"]           # kg eSAF / y
H2 = P["h2_feed"] * 1000 * P["operating_hours"]                   # kg H2 / y
CO2 = P["co2_feed"] * P["operating_hours"]                         # t CO2 / y
F = P["monetary_factor"]
CREDIT = (P["diesel_revenue"] + P["naphtha_revenue"]) * 1e6 * F / Q  # A$/kg
PERF0 = P["load_electrolysis"] / P["h2_feed"]                      # kWh/kg H2 (MW / t/h)


def crf(r, n=P["economic_life"]):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def multiplier(ratio, lr):
    """Experience-curve multiplier (X/X0)^(-b), b = -ln(1-LR)/ln 2 (Eqs. 7-8)."""
    return ratio ** (math.log(1 - lr) / math.log(2))


def auxiliary_mw(b: Boundary):
    if b.legacy_rounded_residual:
        aux = P["legacy_gross_load"] - P["load_electrolysis"]
    else:
        aux = (P["load_electrolyser_cooling"] + P["load_syngas_compression"]
               + P["load_psa_co2_capture"] + P["load_co2_compression"])
    if b.procured_co2:
        aux -= P["load_psa_co2_capture"] + P["load_co2_compression"]
    if b.credit_pem:
        aux -= P["pem_recovery"]
    return aux


def state(year, elec, wacc, ael_lr, ft_lr, b: Boundary = HEADLINE, bop_local_units=None):
    """Annual plant-gate cost state (Eqs. 1-8). Returns A$/kg blocks and net cost."""
    e = EXP[year]
    ael_ratio = float(e["electrolysis_GW"]) / P["ael_reference_experience"]
    co2_ratio = float(e["point_capture_Mt_per_y"]) / P["co2_reference_experience"]
    ft_ratio = float(e["ft_efuel_equivalent_units"]) / P["ft_reference_experience"]
    ael_unit = max(P["ael_cost_floor"], P["ael_installed_cost"] * F * multiplier(ael_ratio, ael_lr))
    k_el = P["load_electrolysis"] * 1000 * ael_unit
    k_syn = P["synthesis_capex"] * 1e6 * F * max(P["synthesis_cost_floor"], multiplier(ft_ratio, ft_lr))
    k_bop = P["bop_capex"] * 1e6 * F if b.include_bop else 0.0
    if bop_local_units is not None and b.include_bop:  # 5% local BOP learning diagnostic (Section 2.7)
        k_bop *= max(P["bop_cost_floor"], multiplier(bop_local_units, P["bop_local_learning_diagnostic"]))
    K = k_el + k_syn + k_bop
    intensity = max(P["ael_performance_floor"], PERF0 * multiplier(ael_ratio, P["ael_performance_improvement"]))
    mwh = H2 * intensity / 1000 + auxiliary_mw(b) * P["operating_hours"]
    price = float(ELEC[year][f"{elec.lower()}_AUD_per_MWh"])
    co2_price = max(P["co2_price_floor"], P["co2_price"] * multiplier(co2_ratio, P["co2_learning_rate"]))
    blocks = {"capital": K * crf(wacc) / Q, "electricity": mwh * price / Q,
              "co2": CO2 * co2_price / Q, "fixed_om": K * P["fixed_om_fraction"] / Q,
              "coproduct_credit": -CREDIT}
    gross = {k: v for k, v in blocks.items() if k != "coproduct_credit"}
    return {"year": year, "electricity_path": elec, "wacc": wacc, "ael_learning_rate": ael_lr,
            "ft_learning_rate": ft_lr, **blocks, "net_cost": sum(blocks.values()),
            "largest_gross_block": max(gross, key=gross.get), "electrolyser_kWh_per_kgH2": intensity,
            "gross_electricity_MWh_per_kg": mwh / Q, "installed_capital_AUD": K}


def run(b: Boundary = HEADLINE):
    """All 81 cases x 26 years (2,106 case-year states)."""
    return [state(y, *c, b=b) for y in YEARS for c in itertools.product(*FACTORS)]


def factorial_shares(costs):
    """Balanced 3^4 decomposition of between-case sum of squares (Section 2.6).
    costs: dict {(elec, wacc, ael, ft): cost}. Returns shares (%) of all 15 effects and groups."""
    import numpy as np
    arr = np.array([[[[costs[(a, b, c, d)] for d in FACTORS[3]] for c in FACTORS[2]]
                     for b in FACTORS[1]] for a in FACTORS[0]])
    g = arr.mean(); total = ((arr - g) ** 2).sum()
    names = ["electricity", "wacc", "ael", "ft"]; eff = {}
    for k in range(1, 5):
        for sub in itertools.combinations(range(4), k):
            other = tuple(i for i in range(4) if i not in sub)
            e = (arr.mean(axis=other, keepdims=True) if other else arr) - g
            for k2 in range(1, k):
                for s2 in itertools.combinations(sub, k2):
                    e = e - eff[s2]
            eff[sub] = np.broadcast_to(e, arr.shape)
    ss = {s: float((v ** 2).sum()) for s, v in eff.items()}
    share = {" x ".join(names[i] for i in s): (100 * v / total if total > 0 else 0.0) for s, v in ss.items()}
    local = {(0,), (1,)}; glob = {(2,), (3,)}
    share["GROUP local main effects"] = sum(100 * ss[s] / total for s in local) if total else 100.0
    share["GROUP global-learning main effects"] = sum(100 * ss[s] / total for s in glob) if total else 0.0
    share["GROUP global-local interactions"] = (sum(100 * v / total for s, v in ss.items()
        if any(i in s for i in (0, 1)) and any(i in s for i in (2, 3))) if total else 0.0)
    return share


def timing_diagnostic(year, b: Boundary = HEADLINE):
    """Delayed-minus-early unit cost (Eq. 9), central case, without and with 5% local BOP learning."""
    d = DOM[year]
    base = state(year, *CENTRAL, b=b)["net_cost"]
    early = state(year, *CENTRAL, b=b, bop_local_units=float(d["early_units"]))["net_cost"]
    delayed = state(year, *CENTRAL, b=b, bop_local_units=float(d["delayed_units"]))["net_cost"]
    return {"year": year, "no_local_learning": base - base, "local_bop_5pct": delayed - early,
            "early_cost_local": early, "delayed_cost_local": delayed}
