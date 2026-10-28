"""Assert that the model reproduces every headline value reported in the manuscript (two-decimal rounding)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import esaf_model as m

r2 = lambda x: round(float(x), 2)
rows = m.run()
c = {y: m.state(y, *m.CENTRAL) for y in m.YEARS}
c50 = np.array([r["net_cost"] for r in rows if r["year"] == 2050])
sh = lambda y: m.factorial_shares({(r["electricity_path"], r["wacc"], r["ael_learning_rate"], r["ft_learning_rate"]): r["net_cost"] for r in rows if r["year"] == y})
s25, s30, s40, s50 = sh(2025), sh(2030), sh(2040), sh(2050)
checks = {
    "central 2025/2030/2040/2050": ([r2(c[y]["net_cost"]) for y in (2025, 2030, 2040, 2050)], [11.91, 9.04, 7.50, 6.70]),
    "2050 P10/median/P90": ([r2(np.percentile(c50, q)) for q in (10, 50, 90)], [5.49, 7.06, 8.87]),
    "2050 min/max": ([r2(c50.min()), r2(c50.max())], [4.72, 10.64]),
    "2050 electricity-largest cases": ([sum(r["largest_gross_block"] == "electricity" for r in rows if r["year"] == 2050)], [78]),
    "2025 blocks cap/elec/co2/fom/credit": ([r2(c[2025][k]) for k in ("capital", "electricity", "co2", "fixed_om", "coproduct_credit")], [3.99, 7.04, 0.91, 1.48, -1.50]),
    "2030 blocks cap/elec/co2/fom": ([r2(c[2030][k]) for k in ("capital", "electricity", "co2", "fixed_om")], [2.88, 5.72, 0.88, 1.07]),
    "2050 elec/capital": ([r2(c[2050]["electricity"]), r2(c[2050]["capital"])], [4.46, 2.13]),
    "2030 local/global": ([r2(s30["GROUP local main effects"]), r2(s30["GROUP global-learning main effects"])], [87.59, 11.90]),
    "2040 local/global": ([r2(s40["GROUP local main effects"]), r2(s40["GROUP global-learning main effects"])], [78.95, 20.19]),
    "2050 local/global/cross": ([r2(s50[k]) for k in ("GROUP local main effects", "GROUP global-learning main effects", "GROUP global-local interactions")], [70.35, 28.44, 1.21]),
    "2050 elec/wacc/ael/ft/waccxael/waccxft": ([r2(s50[k]) for k in ("electricity", "wacc", "ael", "ft", "wacc x ael", "wacc x ft")], [47.11, 23.24, 27.09, 1.35, 1.15, 0.06]),
    "2025 elec/wacc": ([r2(s25["electricity"]), r2(s25["wacc"])], [58.77, 41.23]),
    "timing 5% local BOP 2030/2040/2050": ([r2(m.timing_diagnostic(y)["local_bop_5pct"]) for y in (2030, 2040, 2050)], [0.10, 0.06, 0.02]),
}
ok = True
for name, (got, want) in checks.items():
    status = "PASS" if got == want else "FAIL"; ok &= got == want
    print(f"{status}  {name}: {got}")
assert ok, "a reported value is not reproduced"
print("All reported headline values reproduced.")
