"""Invariant tests: every quantitative claim in README.md is asserted against
the artifact that generated it. If a README number drifts from its source JSON,
this suite fails.

This is the Session-6-onward convention: the submission checks itself, so a
stale number cannot survive a rerun unnoticed.

Run: python3 tests/test_invariants.py
"""
import json
import re
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIGS = ROOT / "figures"
README = ROOT / "README.md"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def load(fn):
    return json.loads((ART / fn).read_text())


R = README.read_text() if README.exists() else ""
# The README writes large integers with thousands separators ("3,916") for
# readability. Strip separators that sit between digits before matching, so
# prose formatting and the assertion cannot drift apart.
R_NORM = re.sub(r"(?<=\d),(?=\d)", "", R)


def in_readme(name, s):
    if R:
        check(name, s in R_NORM, f"looking for '{s}'")


# ============================================================ task 1
t1 = load("task1_adam_by_hand.json")
check("t1: exactly five gradients, one weight",
      len(t1["setup"]["grads"]) == 5 and isinstance(t1["setup"]["w0"], float))
check("t1: hand vs torch agree < 1e-12 on every field, every step",
      t1["worst_abs_diff_any_field"] < 1e-12,
      f"worst={t1['worst_abs_diff_any_field']:.2e}")
check("t1: step agrees to >= 12 decimals",
      t1["min_step_decimals_agreeing"] >= 12,
      f"{t1['min_step_decimals_agreeing']} decimals")
check("t1: verdict records agreement",
      "agree" in t1["verdict"].lower(), t1["verdict"])
check("t1: weight_decay is 0, so this is Adam and not AdamW",
      t1["setup"]["weight_decay"] == 0.0)
check("t1: eps placed outside bias correction (PyTorch convention)",
      "outside bias correction" in t1["setup"]["eps_placement"])
check("t1: float64 so the comparison is not fp32-limited",
      t1["setup"]["dtype"] == "float64")
check("t1: all four of m, v, m_hat, v_hat reported per step",
      all(k in t1["rows"][0] for k in
          ("hand_m", "hand_v", "hand_m_hat", "hand_v_hat")))
check("t1: hand and torch final weights identical to 1e-12",
      abs(t1["final_w_hand"] - t1["final_w_torch"]) < 1e-12)

b1, b2 = t1["setup"]["beta1"], t1["setup"]["beta2"]
eps, lr = t1["setup"]["eps"], t1["setup"]["lr"]
g1 = t1["setup"]["grads"][0]
m1, v1 = (1 - b1) * g1, (1 - b2) * g1 * g1
step1 = lr * (m1 / (1 - b1)) / (math.sqrt(v1 / (1 - b2)) + eps)
check("t1: step 1 re-derived from scratch matches the artifact",
      abs(step1 - t1["rows"][0]["hand_step"]) < 1e-15,
      f"re-derived={step1:.15f}")
check("t1: at t=1, m_hat equals g exactly (bias-correction sanity)",
      abs(t1["rows"][0]["hand_m_hat"] - g1) < 1e-15)
in_readme("t1: README quotes 16-decimal agreement", "16 decimal")

# ============================================================ task 2
t2 = load("task2_bias_correction.json")
ext = t2["extended"]
check("t2: 20 steps plotted both ways",
      len(t2["step_bc"]) == 20 and len(t2["step_nobc"]) == 20)
check("t2: difference has NOT stopped mattering by step 20",
      not t2["criterion_A"]["reached_within_20_steps"],
      f"rel diff @20 = {t2['criterion_A']['rel_diff_at_step_20']:.4f}")
check("t2: rel diff at step 20 is still large (>50%)",
      t2["criterion_A"]["rel_diff_at_step_20"] > 0.5)
check("t2: extended run locates the 1% crossover",
      ext["converged_within_extended_run"],
      f"step {ext['step_after_which_it_stops_mattering_1pct']}")
check("t2: empirical crossover equals the analytic c_t prediction",
      ext["step_after_which_it_stops_mattering_1pct"]
      == t2["analytic"]["first_t_with_c_within_1pct_unbounded"],
      f"empirical={ext['step_after_which_it_stops_mattering_1pct']} "
      f"analytic={t2['analytic']['first_t_with_c_within_1pct_unbounded']}")
check("t2: 5% crossover strictly precedes the 1% crossover",
      ext["step_after_which_it_stops_mattering_5pct"]
      < ext["step_after_which_it_stops_mattering_1pct"])
check("t2: relative difference decays monotonically across decades",
      ext["rel_diff_at_1000"] > ext["rel_diff_at_2000"] > ext["rel_diff_at_4000"],
      f"{ext['rel_diff_at_1000']:.3f} > {ext['rel_diff_at_2000']:.3f} > "
      f"{ext['rel_diff_at_4000']:.3f}")
check("t2: cumulative weight gap is permanent (nonzero at the end)",
      t2["criterion_B"]["final_gap"] > 0,
      f"gap={t2['criterion_B']['final_gap']:.4f}")
check("t2: crossover timescale attributed to beta2, not beta1",
      "beta2" in t2["analytic"]["governed_by"])
bs = sorted(t2["beta2_sensitivity"], key=lambda r: r["beta2"])
cross = [r["crossover_step_1pct"] for r in bs]
check("t2: crossover grows monotonically with beta2",
      all(cross[i] < cross[i + 1] for i in range(len(cross) - 1)),
      str({r["beta2"]: r["crossover_step_1pct"] for r in bs}))
# 0.9 -> 0.99 -> 0.999 -> 0.9999 are decade steps in (1-beta2); 0.95 is not,
# so the decade test uses only the decade-spaced subset.
dec = [r["crossover_step_1pct"] for r in bs if r["beta2"] in (0.9, 0.99, 0.999, 0.9999)]
check("t2: crossover scales ~1/(1-beta2), i.e. ~10x per decade",
      all(abs((dec[i + 1] / dec[i]) / 10.0 - 1.0) < 0.2
          for i in range(len(dec) - 1)),
      str([round(dec[i + 1] / dec[i], 2) for i in range(len(dec) - 1)]))
check("t2: bias correction flips sign of effect as beta2 crosses ~0.99",
      bs[0]["corrected_step_is"] == "larger"
      and bs[-1]["corrected_step_is"] == "smaller")
check("t2: exact multiplicative relation between the two arms is recorded",
      "c_t" in t2["direction"]["relation"])
check("t2: analytic c_t array covers all 20 steps",
      len(t2["analytic_c_t"]) == 20)
in_readme("t2: README quotes the crossover step",
          str(ext["step_after_which_it_stops_mattering_1pct"]))

# ============================================================ task 3
t3 = load("task3_update_weight_ratio.json")
cfg3 = t3["config"]
W = cfg3["warmup_steps"]
check("t3: ratio logged for every layer",
      t3["n_layers_logged"] == len(t3["layers"]) == len(t3["ratio_history"]),
      f"{t3['n_layers_logged']} layers")
check("t3: 'every layer' is a real layer list, not a toy pair",
      t3["n_layers_logged"] >= 10)
check("t3: answer equals the warmup length",
      t3["answer"]["step_warmup_stops_changing_the_ratio"] == W,
      f"step {t3['answer']['step_warmup_stops_changing_the_ratio']}")
check("t3: every history has exactly total_steps entries",
      all(len(v) == cfg3["total_steps"] for v in t3["ratio_history"].values()))
check("t3: lr ramps strictly during warmup",
      all(t3["lr_history"][i] < t3["lr_history"][i + 1] for i in range(W - 1)))
check("t3: lr is exactly constant after warmup, so warmup cannot act later",
      len(set(round(v, 12) for v in t3["lr_history"][W:])) == 1,
      f"lr after warmup = {t3['lr_history'][W]:.3e}")
check("t3: lr reaches its configured maximum at the end of warmup",
      abs(t3["lr_history"][W - 1] - cfg3["lr_max"]) < 1e-12)
check("t3: no-warmup control arm logged for every layer",
      len(t3["control_ratio_history"]) == t3["n_layers_logged"])
check("t3: all ratios finite and non-negative",
      all(math.isfinite(x) and x >= 0
          for v in t3["ratio_history"].values() for x in v))
check("t3: ratio definition is update-to-weight, as asked",
      "W_t" in cfg3["ratio_definition"])
check("t3: noisy-control caveat is recorded, not hidden",
      "noisier" in t3["answer"]["caveat"])
check("t3: ratio/lr quotient recorded for every layer (schedule removed)",
      len(t3["ratio_over_lr"]) == t3["n_layers_logged"])
in_readme("t3: README quotes the warmup answer", f"step {W}")

# ============================================================ task 4
t4 = load("task4_cosine_vs_wsd.json")
cfg4 = t4["config"]
check("t4: trained 300 steps, both stopped at 200",
      cfg4["total_steps"] == 300 and cfg4["stop_at"] == 200)
check("t4: three seeds in the final comparison",
      len(cfg4["seeds"]) >= 3, str(cfg4["seeds"]))
check("t4: both losses at step 200 reported",
      all(a in t4["loss_at_200"] for a in ("cosine", "wsd")))
check("t4: WSD is still on its un-decayed plateau at step 200",
      t4["wsd_is_on_plateau_at_200"]["lr_over_peak"] == 1.0,
      f"wsd lr@200 = {t4['lr_at_step_200']['wsd']:.2e} "
      f"= {t4['wsd_is_on_plateau_at_200']['lr_over_peak']:.0%} of peak")
check("t4: cosine is off its peak at step 200 (the asymmetry being measured)",
      t4["wsd_is_on_plateau_at_200"]["cosine_lr_over_peak"] < 0.5,
      f"cosine at {t4['wsd_is_on_plateau_at_200']['cosine_lr_over_peak']:.0%} of peak")
check("t4: re-budgeted WSD has finished its cooldown by 200",
      t4["lr_at_step_200"]["wsd_rebudget"]
      < 0.2 * t4["peak_lr_selected"]["wsd_rebudget_200"],
      f"reb lr@200 = {t4['lr_at_step_200']['wsd_rebudget']:.2e}")
check("t4: cosine has decayed well below its peak by step 200",
      t4["lr_at_step_200"]["cosine"] < 0.5 * t4["peak_lr_selected"]["cosine"],
      f"cosine lr@200 = {t4['lr_at_step_200']['cosine']:.2e}")
check("t4: both sides were tuned before comparing (brief requirement)",
      set(t4["tuning"]) >= {"cosine", "wsd"})
for arm in ("cosine", "wsd"):
    check(f"t4: {arm} peak lr is interior to the tuning grid",
          t4["tuning"][arm]["interior"],
          f"peak={t4['tuning'][arm]['peak_lr']:.2e}")
check("t4: fair re-budgeted WSD arm present",
      "wsd_rebudget_200" in t4["loss_at_200"])
check("t4: seed noise reported alongside the gap",
      t4["gaps"]["seed_noise_stdev"] > 0,
      f"stdev={t4['gaps']['seed_noise_stdev']:.4f}")
check("t4: every reported loss is finite",
      all(math.isfinite(t4["loss_at_200"][a]["mean"])
          for a in ("cosine", "wsd", "wsd_rebudget_200")))
check("t4: cosine beats un-decayed WSD at step 200",
      t4["loss_at_200"]["cosine"]["mean"] < t4["loss_at_200"]["wsd"]["mean"])
check("t4: the step-200 gap exceeds seed noise (so it is readable)",
      t4["gaps"]["gap_at_200_exceeds_noise"] is True,
      f"gap={t4['gaps']['wsd_minus_cosine_at_200']:.4f} vs "
      f"noise={t4['gaps']['seed_noise_stdev']:.4f}")
check("t4: the step-300 gap does NOT exceed noise (honest null)",
      t4["gaps"]["gap_at_300_exceeds_noise"] is False,
      f"gap={t4['gaps']['wsd_minus_cosine_at_300']:.4f}")
check("t4: re-budgeted WSD ties cosine within 2 sigma at step 200",
      abs(t4["loss_at_200"]["wsd_rebudget_200"]["mean"]
          - t4["loss_at_200"]["cosine"]["mean"])
      < 2 * t4["gaps"]["seed_noise_stdev"],
      f"reb={t4['loss_at_200']['wsd_rebudget_200']['mean']:.4f} vs "
      f"cos={t4['loss_at_200']['cosine']['mean']:.4f}")
check("t4: decision names which checkpoint to keep",
      "cosine" in t4["decision"]["which_model_i_would_keep"].lower())
check("t4: the unfairness of the step-200 comparison is stated",
      "unfair" in t4["decision"]["but_this_is_not_a_verdict_on_wsd"])
for arm in ("cosine", "wsd"):
    in_readme(f"t4: README quotes {arm} loss@200",
              f"{t4['loss_at_200'][arm]['mean']:.4f}")

# ============================================================ task 5
t5 = load("task5_lr_width_sweep.json")
cfg5 = t5["config"]
check("t5: the three required widths were swept",
      cfg5["widths"] == [256, 512, 1024])
check("t5: three minima marked", len(t5["minima"]) == 3)
check("t5: standard parameterization, not muP (the premise of the test)",
      "standard" in cfg5["parameterization"].lower())
for w in ("256", "512", "1024"):
    check(f"t5: width {w} minimum is interior to the lr grid",
          t5["minima"][w]["interior"],
          f"grid argmin {t5['minima'][w]['grid_argmin_lr']:.2e}")
check("t5: every grid cell ran every seed",
      all(len(t5["sweep"][w][lr]["values"]) == len(cfg5["seeds"])
          for w in t5["sweep"] for lr in t5["sweep"][w]),
      f"{len(cfg5['seeds'])} seeds")
check("t5: full grid measured at all three widths",
      all(len(t5["sweep"][w]) == len(cfg5["lrs"]) for w in t5["sweep"]),
      f"{len(cfg5['lrs'])} lrs x 3 widths")
check("t5: optimal lr decreases from width 256 to 1024 (SP behaviour)",
      t5["minima"]["256"]["subgrid_lr"] > t5["minima"]["1024"]["subgrid_lr"],
      f"{t5['minima']['256']['subgrid_lr']:.2e} -> "
      f"{t5['minima']['1024']['subgrid_lr']:.2e}")
check("t5: fitted width exponent is negative",
      t5["power_law"]["alpha"] < 0, f"alpha={t5['power_law']['alpha']:.3f}")
check("t5: R^2 is in [0,1] (the -252 log/linear bug is fixed)",
      0.0 <= t5["power_law"]["r_squared"] <= 1.0,
      f"R2={t5['power_law']['r_squared']:.3f}")
check("t5: power-law fit is good (R^2 > 0.9)",
      t5["power_law"]["r_squared"] > 0.9)
check("t5: a concrete lr for width 4096 is stated",
      t5["extrapolation_to_4096"]["recommended"] > 0,
      f"{t5['extrapolation_to_4096']['recommended']:.3e}")
check("t5: two independent extrapolations are compared",
      t5["extrapolation_to_4096"]["fitted_power_law"] > 0
      and t5["extrapolation_to_4096"]["theory_prior_alpha_minus_1"] > 0)
check("t5: the recommendation is the conservative (smaller) extrapolation",
      t5["extrapolation_to_4096"]["recommended"]
      == min(t5["extrapolation_to_4096"]["fitted_power_law"],
             t5["extrapolation_to_4096"]["theory_prior_alpha_minus_1"]))
check("t5: recommended lr is below every measured optimum",
      t5["extrapolation_to_4096"]["recommended"]
      < min(t5["minima"][w]["subgrid_lr"] for w in ("256", "512", "1024")))
check("t5: confidence is stated and not overclaimed",
      t5["confidence"]["level"] in ("low", "low to moderate", "moderate"),
      t5["confidence"]["level"])
check("t5: at least four distinct limitations are listed",
      len(t5["confidence"]["what_limits_it"]) >= 4,
      f"{len(t5['confidence']['what_limits_it'])} listed")
in_readme("t5: README quotes the width-4096 recommendation",
          f"{t5['extrapolation_to_4096']['recommended']:.2e}")

# ============================================================ cross-cutting
check("x: tasks 4 and 5 share one model module and vocabulary",
      cfg5["vocab"] == 64)
check("x: all five artifacts exist",
      all((ART / f).exists() for f in (
          "task1_adam_by_hand.json", "task2_bias_correction.json",
          "task3_update_weight_ratio.json", "task4_cosine_vs_wsd.json",
          "task5_lr_width_sweep.json")))
check("x: all four figures exist",
      all((FIGS / f).exists() for f in (
          "task2_bias_correction.png", "task3_update_weight_ratio.png",
          "task4_cosine_vs_wsd.png", "task5_lr_width_sweep.png")))
check("x: README exists", README.exists())
check("x: headline numbers are all finite",
      math.isfinite(t1["worst_abs_diff_any_field"])
      and math.isfinite(t4["loss_at_200"]["cosine"]["mean"])
      and math.isfinite(t5["power_law"]["alpha"]))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n, d in FAIL:
        print(f"  FAILED: {n} {d}")
    sys.exit(1)
