"""Task 2: disable bias correction, plot 20 steps both ways, report the step
after which the difference stops mattering.

"Stops mattering" needs a definition or the answer is vibes. We use two:

  (A) RELATIVE STEP CRITERION (the one we report as the headline)
      |step_bc - step_nobc| / max(|step_bc|, |step_nobc|) < 1%, and it stays
      below 1% for all remaining steps. This is a per-step statement: from this
      step on, the two variants are computing effectively the same update.

  (B) CUMULATIVE WEIGHT CRITERION
      |w_bc - w_nobc| stops growing (per-step growth < 1% of the total gap).
      This never returns to zero: the early steps have already moved the weight
      to a different place and nothing later undoes that. Reported to make the
      distinction explicit, because it is the one people get wrong.

The analytic reason: step_corrected = c_t * step_uncorrected exactly, where
    c_t = (1 - b2^t)^0.5 / (1 - b1^t)
c_t -> 1 as t grows, and the approach is governed by b2, not b1, because
(1-b2^t) carries the long tail (b2=0.999 => a 1000-step timescale) while
(1-b1^t) is essentially 1 after ~40 steps.

DIRECTION OF THE BIAS (an earlier draft of this file had this backwards, so it
is spelled out): with b1=0.9, b2=0.999 we get c_1 = 0.316, i.e. c_t < 1, so the
BIAS-CORRECTED step is the SMALLER one and the uncorrected arm overshoots by
3.16x on step 1. It then gets WORSE before it gets better: c_t falls to a
minimum near t ~ 20-30 (uncorrected step ~6x too large) before climbing back
toward 1 over thousands of steps. The sign of the effect is not universal
either: it flips with b2, because c_1 = sqrt(1-b2)/(1-b1). A b2 sensitivity
table is emitted in the artifact to show the flip.

Constant-gradient regime is used for the analytic check; the reported run uses
a fixed pseudo-random gradient sequence so both arms see identical gradients.
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

W0, LR, B1, B2, EPS = 0.3, 0.1, 0.9, 0.999, 1e-8
N_STEPS = 20
REL_TOL = 0.01

# Deterministic gradient sequence, same for both arms. Mean-positive with noise,
# which is what a real parameter sees: a signal plus batch noise.
def grad_sequence(n, seed=11):
    import random
    rng = random.Random(seed)
    return [round(0.15 + rng.gauss(0.0, 0.10), 10) for _ in range(n)]


def run(grads, bias_correction):
    w, m, v = W0, 0.0, 0.0
    steps, ws, mh, vh = [], [], [], []
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        if bias_correction:
            m_hat, v_hat = m / (1 - B1**t), v / (1 - B2**t)
        else:
            m_hat, v_hat = m, v
        step = LR * m_hat / (math.sqrt(v_hat) + EPS)
        w -= step
        steps.append(step); ws.append(w); mh.append(m_hat); vh.append(v_hat)
    return dict(steps=steps, w=ws, m_hat=mh, v_hat=vh)


def main():
    grads = grad_sequence(N_STEPS)
    bc = run(grads, True)
    nb = run(grads, False)

    rel = []
    for a, b in zip(bc["steps"], nb["steps"]):
        denom = max(abs(a), abs(b))
        rel.append(abs(a - b) / denom if denom > 0 else 0.0)

    # (A) last step whose rel diff is >= tol; answer is that step + 1
    viol = [t for t, r in enumerate(rel, start=1) if r >= REL_TOL]
    if viol:
        step_A = max(viol) + 1
        converged_A = step_A <= N_STEPS
    else:
        step_A, converged_A = 1, True

    # (B) cumulative weight gap and its per-step growth
    gap = [abs(a - b) for a, b in zip(bc["w"], nb["w"])]
    growth = [gap[0]] + [abs(gap[i] - gap[i - 1]) for i in range(1, len(gap))]
    frac_growth = [g / gap[-1] if gap[-1] > 0 else 0.0 for g in growth]
    viol_B = [t for t, f in enumerate(frac_growth, start=1) if f >= REL_TOL]
    step_B = (max(viol_B) + 1) if viol_B else 1

    # analytic factor c_t on the step under a constant gradient
    c = [((1 - B2**t) ** 0.5) / (1 - B1**t) for t in range(1, N_STEPS + 1)]
    c_within = [t for t, ct in enumerate(c, start=1) if abs(ct - 1.0) < REL_TOL]
    c_step = c_within[0] if c_within else None
    # extend analytic search past 20 to find where c_t truly lands inside 1%
    t_ext, c_ext = 1, None
    while t_ext <= 20000:
        ct = ((1 - B2**t_ext) ** 0.5) / (1 - B1**t_ext)
        if abs(ct - 1.0) < REL_TOL:
            c_ext = t_ext
            break
        t_ext += 1

    # ---- EXTENDED RUN -------------------------------------------------------
    # The 20-step window mandated by the assignment does NOT contain the
    # crossover, so reporting only "it hasn't converged" would be a non-answer.
    # We rerun to 6000 steps with the same generator to locate the actual step.
    N_EXT = 6000
    grads_ext = grad_sequence(N_EXT)
    bc_e, nb_e = run(grads_ext, True), run(grads_ext, False)
    rel_e = []
    for a, b in zip(bc_e["steps"], nb_e["steps"]):
        d = max(abs(a), abs(b))
        rel_e.append(abs(a - b) / d if d > 0 else 0.0)
    viol_e = [t for t, r in enumerate(rel_e, start=1) if r >= REL_TOL]
    step_A_ext = (max(viol_e) + 1) if viol_e else 1
    ext_converged = step_A_ext <= N_EXT
    # 5% tolerance, a looser but defensible reading of "stops mattering"
    viol_e5 = [t for t, r in enumerate(rel_e, start=1) if r >= 0.05]
    step_A_ext5 = (max(viol_e5) + 1) if viol_e5 else 1

    # ---- WHICH ARM IS BIGGER, AND SENSITIVITY TO beta2 ----------------------
    # step_corrected = c_t * step_uncorrected exactly, so c_t < 1 means the
    # uncorrected arm overshoots. c_t is non-monotone: it dips before recovering.
    c_full = [((1 - B2**t) ** 0.5) / (1 - B1**t) for t in range(1, 20001)]
    t_min = min(range(len(c_full)), key=lambda i: c_full[i]) + 1

    def crossover(b1, b2, tol=REL_TOL, tmax=200000):
        """Last step at which c_t is still outside tolerance, plus one.

        NOT the first step inside tolerance: c_t is non-monotone, so it can pass
        through 1 early and leave again. An earlier draft returned the first
        crossing and reported step 1 for beta2=0.99, which is wrong: c_1 = 1.000
        there by exact cancellation, but c_2 = 0.743.
        """
        last_bad = None
        for t in range(1, tmax + 1):
            if abs(((1 - b2**t) ** 0.5) / (1 - b1**t) - 1.0) >= tol:
                last_bad = t
        return (last_bad + 1) if last_bad else 1

    beta2_table = [
        dict(beta2=b2,
             c_1=((1 - b2) ** 0.5) / (1 - B1),
             corrected_step_is=("larger" if ((1 - b2) ** 0.5) / (1 - B1) > 1 else "smaller"),
             crossover_step_1pct=crossover(B1, b2))
        for b2 in (0.9, 0.95, 0.99, 0.999, 0.9999)
    ]

    out = dict(
        config=dict(w0=W0, lr=LR, beta1=B1, beta2=B2, eps=EPS,
                    n_steps=N_STEPS, rel_tol=REL_TOL, grads=grads),
        direction=dict(
            relation="step_corrected = c_t * step_uncorrected (exact)",
            c_1=c_full[0],
            overshoot_factor_step_1=1.0 / c_full[0],
            c_t_minimum=min(c_full),
            c_t_argmin_step=t_min,
            worst_overshoot_factor=1.0 / min(c_full),
            which_arm_is_larger_early="uncorrected (because c_t < 1 for these betas)",
            non_monotone=True,
            exact_cancellation_beta2=1 - (1 - B1) ** 2,
            exact_cancellation_note=(
                "c_1 = sqrt(1-b2)/(1-b1), so c_1 = 1 exactly when "
                "b2 = 1-(1-b1)^2 = 0.99 for b1=0.9: at that beta2 the two bias "
                "corrections cancel on step 1 and the direction of the effect "
                "flips either side of it"
            ),
            note=("the gap widens before it closes: the 20-step window the "
                  "assignment asks for sits inside the widening phase"),
        ),
        beta2_sensitivity=beta2_table,
        extended=dict(
            n_steps=N_EXT,
            step_after_which_it_stops_mattering_1pct=step_A_ext,
            step_after_which_it_stops_mattering_5pct=step_A_ext5,
            converged_within_extended_run=ext_converged,
            rel_diff_at_1000=rel_e[999],
            rel_diff_at_2000=rel_e[1999],
            rel_diff_at_4000=rel_e[3999],
            rel_diff_final=rel_e[-1],
            note="the mandated 20-step window does not contain the crossover",
        ),
        step_bc=bc["steps"], step_nobc=nb["steps"],
        w_bc=bc["w"], w_nobc=nb["w"],
        rel_diff_per_step=rel,
        cumulative_w_gap=gap,
        analytic_c_t=c,
        criterion_A=dict(
            definition="per-step |dstep|/max(|step|) < 1% and stays there",
            step_after_which_it_stops_mattering=step_A,
            reached_within_20_steps=converged_A,
            rel_diff_at_step_20=rel[-1],
        ),
        criterion_B=dict(
            definition="cumulative |w_bc - w_nobc| stops growing (<1% of final gap per step)",
            step_after_which_gap_stops_growing=step_B,
            final_gap=gap[-1],
            note="gap never returns to zero; early steps are permanent",
        ),
        analytic=dict(
            formula="c_t = sqrt(1-b2^t)/(1-b1^t)",
            c_1=c[0], c_20=c[-1],
            first_t_with_c_within_1pct_searched_to_20=c_step,
            first_t_with_c_within_1pct_unbounded=c_ext,
            governed_by="beta2 (0.999 -> ~1000 step timescale), not beta1",
        ),
    )
    (ART / "task2_bias_correction.json").write_text(json.dumps(out, indent=2))

    ts = list(range(1, N_STEPS + 1))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0][0].plot(ts, bc["steps"], "o-", label="bias correction ON")
    ax[0][0].plot(ts, nb["steps"], "s--", label="bias correction OFF")
    ax[0][0].set_title("Adam step size, first 20 steps"); ax[0][0].set_xlabel("step")
    ax[0][0].set_ylabel("update magnitude"); ax[0][0].legend(); ax[0][0].grid(alpha=.3)

    ax[0][1].plot(ts, bc["w"], "o-", label="ON")
    ax[0][1].plot(ts, nb["w"], "s--", label="OFF")
    ax[0][1].set_title("weight trajectory"); ax[0][1].set_xlabel("step")
    ax[0][1].set_ylabel("w"); ax[0][1].legend(); ax[0][1].grid(alpha=.3)

    ax[1][0].semilogy(ts, [max(r, 1e-18) for r in rel], "o-")
    ax[1][0].axhline(REL_TOL, color="r", ls=":", label="1% tolerance")
    if converged_A:
        ax[1][0].axvline(step_A, color="g", ls="--", label=f"stops mattering: step {step_A}")
    ax[1][0].set_title("relative per-step difference"); ax[1][0].set_xlabel("step")
    ax[1][0].set_ylabel("|d| / max(|step|)"); ax[1][0].legend(); ax[1][0].grid(alpha=.3)

    ax[1][1].plot(ts, gap, "o-", color="purple")
    ax[1][1].set_title("cumulative |w_ON - w_OFF| (never returns to 0)")
    ax[1][1].set_xlabel("step"); ax[1][1].set_ylabel("weight gap"); ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(FIG / "task2_bias_correction.png", dpi=130)

    print(f"criterion A (per-step, 1%) : stops mattering after step {step_A} "
          f"(reached within 20: {converged_A}); rel diff at step 20 = {rel[-1]:.4f}")
    print(f"criterion B (cumulative)   : gap stops growing after step {step_B}, "
          f"final gap {gap[-1]:.6f} (permanent)")
    print(f"analytic c_1={c[0]:.4f}  c_20={c[-1]:.4f}  "
          f"first t with c within 1% (unbounded) = {c_ext}")
    print(f"direction: c_1={c_full[0]:.4f} -> uncorrected step is "
          f"{1/c_full[0]:.2f}x too large at t=1; worst at t={t_min} "
          f"({1/min(c_full):.2f}x)")
    for row in beta2_table:
        print(f"  beta2={row['beta2']:<7} c_1={row['c_1']:.3f} "
              f"corrected step {row['corrected_step_is']:<7} "
              f"crossover(1%)={row['crossover_step_1pct']}")
    print(f"EXTENDED to {N_EXT}: stops mattering after step {step_A_ext} (1%), "
          f"{step_A_ext5} (5%); rel diff @1000={rel_e[999]:.4f} "
          f"@2000={rel_e[1999]:.4f} @4000={rel_e[3999]:.4f}")


if __name__ == "__main__":
    main()
