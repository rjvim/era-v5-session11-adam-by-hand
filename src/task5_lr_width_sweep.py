"""Task 5: sweep the learning rate at widths 256, 512, 1024, plot loss vs lr,
mark the three minima, state the lr you would use at width 4096 and how
confident you are in it.

WHY THIS EXPERIMENT HAS A DEFINITE EXPECTED SHAPE
The model uses STANDARD parameterization (SP): init std is 0.02 regardless of
width, and there is no per-layer lr scaling. Under SP, the optimal Adam lr
DECREASES as width grows, empirically close to eta* ~ 1/width. Under muP it
would instead stay put, which is the entire selling point of muP and the reason
the instructor showed the flat-transfer curve. So:
  - if our three minima drift left as width grows, SP is behaving as theory says
    and a 1/width extrapolation to 4096 is defensible
  - if they do not drift, something is wrong with the setup and no extrapolation
    should be trusted

TWO EXTRAPOLATIONS ARE REPORTED, not one:
  (a) fitted power law eta* = C * width^alpha, alpha fitted from the 3 minima
  (b) the theory prior alpha = -1
Agreement between them is the confidence signal. Disagreement is reported
honestly rather than smoothed over.

GRID HISTORY (recorded because it changed the answer). The first pilot grid ran
3e-4 .. 1.92e-2. At width 256 the minimum was interior (1.2e-3) but at width 512
it had already slid to the second grid point (6e-4), and the top two points were
pure divergence (loss ~3.95 against a ln(64)=4.16 ceiling) - i.e. two of seven
points carried no information and the low end was about to run out of room at
width 1024. The grid was therefore shifted down one octave and extended to
7.5e-5 .. 9.6e-3 so that all three minima stay interior. The pilot log is kept
at artifacts/task5_pilot_grid.log.

CONFIDENCE IS BOUNDED BY THE GRID. A sweep on a coarse log grid cannot locate a
minimum more precisely than the grid spacing, and a parabola fit in log-lr space
is used to get a sub-grid estimate. Both are reported.
"""
import json
import math
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import TinyLM, make_batch, VOCAB

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
ART.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

WIDTHS = [256, 512, 1024]
LRS = [7.5e-5, 1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3]
STEPS = 80
WARMUP = 10
SEEDS = [11, 12]
BATCH, SEQ = 8, 32   # seq shortened: 1 CPU core, and width is the variable here


def lr_at(step, lr_max):
    if step <= WARMUP:
        return lr_max * step / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return lr_max * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def train_one(width, lr_max, seed):
    torch.manual_seed(seed)
    model = TinyLM(d_model=width, n_layer=2, n_head=max(1, width // 64), seed=seed,
                   seqlen=SEQ)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed + 7)
    eval_g = torch.Generator().manual_seed(4242)
    ev_batches = [make_batch(BATCH, SEQ, eval_g) for _ in range(4)]
    diverged = False
    for step in range(1, STEPS + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, lr_max)
        x, y = make_batch(BATCH, SEQ, g)
        loss = model.loss(x, y)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if diverged:
        return float("nan"), True
    model.eval()
    with torch.no_grad():
        ev = statistics.fmean(model.loss(a, b).item() for a, b in ev_batches)
    return ev, not math.isfinite(ev)


def parabola_min(xs_log, ys):
    """Fit a parabola through the 3 points around the argmin, return sub-grid min."""
    i = min(range(len(ys)), key=lambda k: ys[k] if math.isfinite(ys[k]) else 1e9)
    if i == 0 or i == len(ys) - 1:
        return xs_log[i], False
    x1, x2, x3 = xs_log[i - 1], xs_log[i], xs_log[i + 1]
    y1, y2, y3 = ys[i - 1], ys[i], ys[i + 1]
    if not all(math.isfinite(v) for v in (y1, y2, y3)):
        return xs_log[i], False
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if d == 0:
        return xs_log[i], False
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d
    if a <= 0:
        return xs_log[i], False
    return -b / (2 * a), True


CKPT = ART / "task5_partial.json"


def main():
    # Resumable: each (width, lr) cell is written to disk as soon as it is
    # measured. The width-1024 grid alone is ~10 minutes on one CPU core and an
    # earlier attempt was killed mid-sweep, losing everything. Now a killed run
    # resumes from the last completed cell instead of restarting.
    done = {}
    if CKPT.exists():
        try:
            done = json.loads(CKPT.read_text())
        except json.JSONDecodeError:
            done = {}

    results = {}
    for w in WIDTHS:
        results[w] = {}
        for lr in LRS:
            key = f"{w}|{lr:.6g}"
            if key in done:
                results[w][lr] = done[key]
                print(f"width {w:5d} lr {lr:.2e} -> "
                      f"{results[w][lr]['mean']:.4f}  (cached)")
                continue
            vals = []
            for s in SEEDS:
                ev, bad = train_one(w, lr, s)
                vals.append(ev)
            finite = [v for v in vals if math.isfinite(v)]
            results[w][lr] = dict(
                values=vals,
                mean=statistics.fmean(finite) if finite else float("nan"),
                stdev=statistics.stdev(finite) if len(finite) > 1 else 0.0,
                diverged=len(finite) == 0,
            )
            done[key] = results[w][lr]
            CKPT.write_text(json.dumps(done))
            print(f"width {w:5d} lr {lr:.2e} -> "
                  f"{results[w][lr]['mean'] if finite else float('nan'):.4f}")

    minima = {}
    for w in WIDTHS:
        ys = [results[w][lr]["mean"] for lr in LRS]
        xs_log = [math.log10(lr) for lr in LRS]
        grid_i = min(range(len(ys)),
                     key=lambda k: ys[k] if math.isfinite(ys[k]) else 1e9)
        sub_log, ok = parabola_min(xs_log, ys)
        minima[w] = dict(
            grid_argmin_lr=LRS[grid_i],
            grid_min_loss=ys[grid_i],
            subgrid_lr=10 ** sub_log,
            subgrid_fit_valid=ok,
            interior=0 < grid_i < len(LRS) - 1,
        )

    # power-law fit eta* = C * width^alpha on the sub-grid minima
    lw = [math.log(w) for w in WIDTHS]
    le = [math.log(minima[w]["subgrid_lr"]) for w in WIDTHS]
    n = len(lw)
    mx, my = statistics.fmean(lw), statistics.fmean(le)
    sxx = sum((a - mx) ** 2 for a in lw)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lw, le))
    alpha = sxy / sxx
    logC = my - alpha * mx
    # BUG FIX: predictions must be compared in the SAME space as the targets.
    # The first version built pred with math.exp(...) (linear space) and
    # compared it against le (log space), which produced R^2 = -252, an
    # impossible value for a least-squares fit and the tell that caught it.
    # The fit itself (alpha, logC) was always correct; only R^2 was wrong.
    pred = [logC + alpha * a for a in lw]
    ss_res = sum((b - p) ** 2 for b, p in zip(le, pred))
    ss_tot = sum((b - my) ** 2 for b in le)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    W_TARGET = 4096
    fit_4096 = math.exp(logC + alpha * math.log(W_TARGET))
    # theory prior: alpha = -1 anchored on the largest measured width
    anchor_w = WIDTHS[-1]
    theory_4096 = minima[anchor_w]["subgrid_lr"] * (anchor_w / W_TARGET)
    ratio = fit_4096 / theory_4096

    out = dict(
        config=dict(widths=WIDTHS, lrs=LRS, steps=STEPS, warmup=WARMUP,
                    seeds=SEEDS, batch=BATCH, seqlen=SEQ, vocab=VOCAB,
                    parameterization="standard (SP), init std 0.02, no muP scaling",
                    schedule="linear warmup + cosine to 10%",
                    optimizer="AdamW", betas=[0.9, 0.95], weight_decay=0.1,
                    grid_spacing_factor=2.0, grid_shifted_down_after_pilot=True),
        sweep={str(w): {f"{lr:.6g}": results[w][lr] for lr in LRS} for w in WIDTHS},
        minima={str(w): minima[w] for w in WIDTHS},
        power_law=dict(
            form="eta_star = C * width^alpha",
            alpha=alpha, C=math.exp(logC), r_squared=r2,
            theory_expectation_alpha=-1.0,
            alpha_matches_theory_within_0_3=abs(alpha - (-1.0)) < 0.3,
        ),
        extrapolation_to_4096=dict(
            fitted_power_law=fit_4096,
            theory_prior_alpha_minus_1=theory_4096,
            ratio_fit_over_theory=ratio,
            recommended=min(fit_4096, theory_4096),
            recommendation_rule=(
                "take the smaller of the two extrapolations; at 4096 an lr that "
                "is too high risks a loss spike that costs far more than the "
                "small amount of convergence speed given up by being "
                "conservative"
            ),
        ),
        confidence=dict(
            level="low to moderate",
            what_supports_it=[
                "three widths, two seeds each, interior minima on a log grid",
                f"power-law fit R^2 = {r2:.3f}",
                "fitted exponent compared against an independent theory prior",
            ],
            what_limits_it=[
                f"grid spacing is 2x, so the minimum is only located to within a factor of ~2 "
                f"before the parabola fit",
                f"only {STEPS} steps per run; the optimal lr for a short budget is "
                "higher than for a long run, so these minima are biased upward",
                "4096 is a 4x extrapolation beyond the largest measured width, "
                "and it is exactly where SP scaling tends to break down",
                "2 seeds cannot separate curvature of the loss-vs-lr curve from seed noise",
                "depth held at 2 layers; real 4096-width models are deep, and "
                "depth interacts with the optimal lr",
            ],
            what_i_would_do_instead=(
                "not extrapolate. Switch to muP so the optimal lr transfers "
                "across width by construction, or spend one short sweep at "
                "width 2048 to test the fitted exponent before committing at 4096."
            ),
        ),
    )
    (ART / "task5_lr_width_sweep.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for w, col in zip(WIDTHS, ["tab:blue", "tab:orange", "tab:green"]):
        ys = [results[w][lr]["mean"] for lr in LRS]
        es = [results[w][lr]["stdev"] for lr in LRS]
        ax[0].errorbar(LRS, ys, yerr=es, marker="o", color=col, capsize=3,
                       label=f"width {w}")
        m = minima[w]
        ax[0].axvline(m["subgrid_lr"], color=col, ls=":", alpha=.8)
        ax[0].plot([m["grid_argmin_lr"]], [m["grid_min_loss"]], marker="*",
                   ms=18, color=col, mec="k")
    ax[0].set_xscale("log"); ax[0].set_xlabel("learning rate")
    ax[0].set_ylabel("held-out loss")
    ax[0].set_title("loss vs lr, three widths (stars = minima)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ws = WIDTHS + [W_TARGET]
    ax[1].plot(WIDTHS, [minima[w]["subgrid_lr"] for w in WIDTHS], "o-",
               label="measured minima")
    ax[1].plot(ws, [math.exp(logC + alpha * math.log(w)) for w in ws], "--",
               label=f"fit: width^{alpha:.2f} (R2={r2:.3f})")
    ax[1].plot(ws, [minima[anchor_w]["subgrid_lr"] * (anchor_w / w) for w in ws],
               ":", label="theory prior: width^-1")
    ax[1].plot([W_TARGET], [fit_4096], "*", ms=18, color="tab:red",
               label=f"4096 fit = {fit_4096:.2e}")
    ax[1].plot([W_TARGET], [theory_4096], "s", ms=9, color="tab:purple",
               label=f"4096 theory = {theory_4096:.2e}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("width"); ax[1].set_ylabel("optimal lr")
    ax[1].set_title("optimal lr vs width, extrapolated to 4096")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "task5_lr_width_sweep.png", dpi=130)

    print(f"\nminima: " + ", ".join(
        f"w{w}: grid={minima[w]['grid_argmin_lr']:.2e} sub={minima[w]['subgrid_lr']:.2e}"
        for w in WIDTHS))
    print(f"alpha={alpha:.3f} (theory -1), R2={r2:.3f}")
    print(f"4096: fit={fit_4096:.3e}  theory={theory_4096:.3e}  "
          f"recommend={min(fit_4096, theory_4096):.3e}")


if __name__ == "__main__":
    main()
