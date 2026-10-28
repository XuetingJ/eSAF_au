"""Run the headline model and accounting-boundary tests; write all results used in the paper to outputs/."""
import csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import esaf_model as m

OUT = m.ROOT / "outputs"; OUT.mkdir(exist_ok=True)


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarise(b):
    rows = m.run(b)
    by_year = {y: [r for r in rows if r["year"] == y] for y in m.YEARS}
    central = {y: m.state(y, *m.CENTRAL, b=b) for y in m.YEARS}
    stats, shares = [], []
    for y, rs in by_year.items():
        c = np.array([r["net_cost"] for r in rs])
        stats.append({"year": y, "central": central[y]["net_cost"], "min": c.min(), "p10": np.percentile(c, 10),
                      "median": np.percentile(c, 50), "p90": np.percentile(c, 90), "max": c.max(),
                      "n_electricity_largest": sum(r["largest_gross_block"] == "electricity" for r in rs),
                      "n_capital_largest": sum(r["largest_gross_block"] == "capital" for r in rs)})
        sh = m.factorial_shares({(r["electricity_path"], r["wacc"], r["ael_learning_rate"], r["ft_learning_rate"]): r["net_cost"] for r in rs})
        shares.append({"year": y, **sh})
    return rows, central, stats, shares


def marginal(rows, year, factor_idx, level, fixed=None):
    keys = ["electricity_path", "wacc", "ael_learning_rate", "ft_learning_rate"]
    sel = [r for r in rows if r["year"] == year and r[keys[factor_idx]] == level
           and all(r[keys[i]] == v for i, v in (fixed or {}).items())]
    return float(np.mean([r["net_cost"] for r in sel]))


if __name__ == "__main__":
    rows, central, stats, shares = summarise(m.HEADLINE)
    write("case_year_costs.csv", rows)
    write("central_path.csv", [central[y] for y in m.YEARS])
    write("annual_statistics.csv", stats)
    write("factorial_shares.csv", shares)
    timing = [m.timing_diagnostic(y) for y in m.YEARS]
    write("timing_diagnostic.csv", timing)
    # marginal means and learning benefits, 2050
    mm = []
    for i, levels in enumerate(m.FACTORS):
        for lv in levels:
            mm.append({"factor": ["electricity_path", "wacc", "ael_learning_rate", "ft_learning_rate"][i],
                       "level": lv, "marginal_mean_2050": marginal(rows, 2050, i, lv)})
    write("marginal_means_2050.csv", mm)
    benefit = []
    for w in m.FACTORS[1]:
        benefit.append({"wacc": w,
            "ael_5_to_18": marginal(rows, 2050, 2, 0.05, {1: w}) - marginal(rows, 2050, 2, 0.18, {1: w}),
            "ft_0_to_10": marginal(rows, 2050, 3, 0.0, {1: w}) - marginal(rows, 2050, 3, 0.10, {1: w})})
    write("learning_benefit_by_wacc_2050.csv", benefit)
    # boundary tests
    tests = {"Headline: procured CO2": m.HEADLINE,
             "PEM recovery credited (-1.95 MW)": m.Boundary(credit_pem=True),
             "BOP allowance removed (-US$80 million)": m.Boundary(include_bop=False),
             "Legacy accounting: capture and CO2-compression loads retained": m.LEGACY}
    bt = []
    for name, b in tests.items():
        _, c, st, sh = summarise(b)
        s50 = next(s for s in st if s["year"] == 2050); h50 = next(s for s in sh if s["year"] == 2050)
        bt.append({"test": name, "central_2025": c[2025]["net_cost"], "central_2050": c[2050]["net_cost"],
                   "p10_2050": s50["p10"], "p90_2050": s50["p90"],
                   "local_2050": h50["GROUP local main effects"], "global_2050": h50["GROUP global-learning main effects"],
                   "cross_2050": h50["GROUP global-local interactions"], "n_electricity_largest_2050": s50["n_electricity_largest"]})
    write("boundary_tests.csv", bt)
    print("Results written to", OUT)
