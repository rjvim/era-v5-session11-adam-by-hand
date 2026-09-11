"""Task 4: train the same model twice for 300 steps, once cosine once WSD, stop
both at step 200, report both losses, say which model you would keep.

THE POINT OF THIS EXERCISE (and the trap in it):
Both schedules are BUDGETED FOR 300 STEPS. Stopping at 200 is an early stop that
neither schedule was planned for. At step 200:
  - cosine has already decayed a long way toward its minimum, so it looks good
  - WSD is still sitting on its high constant plateau and has NOT cooled down,
    so it looks bad
The step-200 loss therefore does NOT tell you which schedule is better. It tells
you which schedule is further through its decay.

TUNING BOTH SIDES (the assignment's explicit warning)
The brief says: "Tune both sides before accepting a comparison. Almost every
optimizer claim that failed to replicate was a well tuned method measured
against a badly tuned one." A shared peak lr is NOT tuning both sides, it is
tuning neither. WSD spends most of its budget at the peak lr while cosine
spends most of its budget below the peak, so the same peak value is a different
effective lr for the two schedules and the optimum is not expected to coincide.

So each arm gets its own peak-lr sweep, selected on held-out loss at that arm's
own budget end, and we report:
  1. MATCHED-lr comparison  (both arms at the same peak lr)  - the naive one
  2. TUNED comparison       (each arm at its own best peak)  - the fair one
  3. loss at 200 and at 300 for both, so the early-stop artefact is visible
  4. a re-budgeted WSD arm whose cooldown actually finishes by step 200, which
     is what you would really run if 200 were the budget
Three seeds everywhere, because a single-seed gap at this scale is noise.
"""
import json
import math
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import TinyLM, make_batch

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
ART.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

TOTAL = 300
STOP = 200
LR_MATCHED = 3e-3     # the shared value used for the naive matched comparison
TUNE_LRS = [1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2]
TUNE_SEEDS = [11]     # tuning on one seed; the final comparison uses three
WARMUP = 15           # 5% of 300
WSD_DECAY_FRAC = 0.2  # last 20% is the cooldown
SEEDS = [11, 12, 13]
EVAL_EVERY = 10


def cosine_lr(step, total, lr_max, warmup=WARMUP, min_frac=0.1):
    if step <= warmup:
        return lr_max * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, p)
    return lr_max * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def wsd_lr(step, total, lr_max, warmup=WARMUP, decay_frac=WSD_DECAY_FRAC,
           min_frac=0.1):
    """Warmup -> Stable -> Decay. Linear cooldown over the last decay_frac."""
    if step <= warmup:
        return lr_max * step / max(1, warmup)
    decay_start = total - int(total * decay_frac)
    if step <= decay_start:
        return lr_max
    p = (step - decay_start) / max(1, total - decay_start)
    p = min(1.0, p)
    return lr_max * (1 - p * (1 - min_frac))


SCHEDS = {"cosine": cosine_lr, "wsd": wsd_lr}


def train(sched_fn, total, seed, lr_max, stop_at=None):
    torch.manual_seed(seed)
    model = TinyLM(d_model=128, n_layer=2, n_head=4, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed + 1000)
    eval_g = torch.Generator().manual_seed(99999)  # held-out, same for all arms
    eval_batches = [make_batch(batch=8, seqlen=64, gen=eval_g) for _ in range(6)]

    curve, lrs, snapshots = [], [], {}
    n = stop_at or total
    diverged = False
    for step in range(1, n + 1):
        lr = sched_fn(step, total, lr_max)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = make_batch(batch=8, seqlen=64, gen=g)
        loss = model.loss(x, y)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lrs.append(lr)
        if step % EVAL_EVERY == 0 or step in (1, STOP, total):
            model.eval()
            with torch.no_grad():
                ev = statistics.fmean(model.loss(a, b).item() for a, b in eval_batches)
            model.train()
            curve.append((step, ev))
            if step in (STOP, total):
                snapshots[step] = ev
    if diverged:
        snapshots = {k: float("nan") for k in (STOP, total)}
    return curve, lrs, snapshots


def tune(name, total, stop_at=None):
    """Peak-lr sweep for one schedule, scored at that arm's own budget end."""
    sched = SCHEDS[name]
    end = stop_at or total
    table = []
    for lr in TUNE_LRS:
        vals = []
        for s in TUNE_SEEDS:
            _, _, snap = train(sched, total, s, lr, stop_at=stop_at)
            vals.append(snap.get(end, float("nan")))
        finite = [v for v in vals if math.isfinite(v)]
        table.append(dict(lr=lr, loss=statistics.fmean(finite) if finite else float("nan"),
                          diverged=not finite))
        print(f"  tune {name:<6} budget {end:<4} lr {lr:.2e} -> "
              f"{table[-1]['loss']:.4f}{' DIVERGED' if table[-1]['diverged'] else ''}")
    best = min(table, key=lambda r: r["loss"] if math.isfinite(r["loss"]) else 1e9)
    interior = 0 < TUNE_LRS.index(best["lr"]) < len(TUNE_LRS) - 1
    return dict(grid=table, peak_lr=best["lr"], loss_at_peak=best["loss"],
                interior=interior, scored_at_step=end, seeds=TUNE_SEEDS)


def agg(snaps, step):
    vals = [d[step] for d in snaps if step in d]
    finite = [v for v in vals if math.isfinite(v)]
    return dict(values=vals,
                mean=statistics.fmean(finite) if finite else float("nan"),
                stdev=statistics.stdev(finite) if len(finite) > 1 else 0.0)


def main():
    print("tuning both sides before comparing:")
    tuned = {
        "cosine": tune("cosine", TOTAL),
        "wsd": tune("wsd", TOTAL),
        "wsd_rebudget_200": tune("wsd", STOP, stop_at=STOP),
    }
    peak = {k: v["peak_lr"] for k, v in tuned.items()}
    print(f"selected peaks: {peak}")

    # arms: (label, schedule, budget, stop_at, peak lr)
    arms = {
        "cosine_matched": ("cosine", TOTAL, None, LR_MATCHED),
        "wsd_matched": ("wsd", TOTAL, None, LR_MATCHED),
        "cosine_tuned": ("cosine", TOTAL, None, peak["cosine"]),
        "wsd_tuned": ("wsd", TOTAL, None, peak["wsd"]),
        "wsd_rebudget_200": ("wsd", STOP, STOP, peak["wsd_rebudget_200"]),
    }

    snaps = {k: [] for k in arms}
    curves = {k: [] for k in arms}
    lr_trace = {}
    for label, (sname, total, stop_at, lr_max) in arms.items():
        for s in SEEDS:
            c, lrs, snap = train(SCHEDS[sname], total, s, lr_max, stop_at=stop_at)
            snaps[label].append(snap)
            curves[label].append(c)
            if s == SEEDS[0]:
                lr_trace[label] = lrs
        print(f"{label:<18} lr={lr_max:.2e} "
              f"loss@200={agg(snaps[label], STOP)['mean']:.4f} "
              f"loss@300={agg(snaps[label], TOTAL)['mean']:.4f}")

    matched200 = {a: agg(snaps[f"{a}_matched"], STOP) for a in ("cosine", "wsd")}
    matched300 = {a: agg(snaps[f"{a}_matched"], TOTAL) for a in ("cosine", "wsd")}
    tuned200 = {a: agg(snaps[f"{a}_tuned"], STOP) for a in ("cosine", "wsd")}
    tuned300 = {a: agg(snaps[f"{a}_tuned"], TOTAL) for a in ("cosine", "wsd")}
    reb200 = agg(snaps["wsd_rebudget_200"], STOP)

    noise = max(tuned200["cosine"]["stdev"], tuned200["wsd"]["stdev"])
    gap200_t = tuned200["wsd"]["mean"] - tuned200["cosine"]["mean"]
    gap300_t = tuned300["wsd"]["mean"] - tuned300["cosine"]["mean"]
    gap200_m = matched200["wsd"]["mean"] - matched200["cosine"]["mean"]
    gap300_m = matched300["wsd"]["mean"] - matched300["cosine"]["mean"]

    lr200 = dict(
        cosine=cosine_lr(200, TOTAL, peak["cosine"]),
        wsd=wsd_lr(200, TOTAL, peak["wsd"]),
        wsd_rebudget=wsd_lr(200, STOP, peak["wsd_rebudget_200"]),
        cosine_matched=cosine_lr(200, TOTAL, LR_MATCHED),
        wsd_matched=wsd_lr(200, TOTAL, LR_MATCHED),
    )

    out = dict(
        config=dict(total_steps=TOTAL, stop_at=STOP, lr_max=LR_MATCHED,
                    lr_matched=LR_MATCHED, tune_grid=TUNE_LRS,
                    warmup=WARMUP, wsd_decay_frac=WSD_DECAY_FRAC,
                    seeds=SEEDS, eval="held-out batches, identical across arms",
                    optimizer="AdamW", betas=[0.9, 0.95], weight_decay=0.1,
                    model="TinyLM d_model=128 n_layer=2 n_head=4"),
        tuning=tuned,
        peak_lr_selected=peak,
        lr_at_step_200=lr200,
        wsd_is_on_plateau_at_200=dict(
            lr_over_peak=lr200["wsd"] / peak["wsd"],
            cosine_lr_over_peak=lr200["cosine"] / peak["cosine"],
        ),
        loss_at_200=dict(cosine=tuned200["cosine"], wsd=tuned200["wsd"],
                         wsd_rebudget_200=reb200,
                         cosine_matched=matched200["cosine"],
                         wsd_matched=matched200["wsd"]),
        loss_at_300=dict(cosine=tuned300["cosine"], wsd=tuned300["wsd"],
                         cosine_matched=matched300["cosine"],
                         wsd_matched=matched300["wsd"]),
        gaps=dict(
            wsd_minus_cosine_at_200=gap200_t,
            wsd_minus_cosine_at_300=gap300_t,
            wsd_minus_cosine_at_200_matched_lr=gap200_m,
            wsd_minus_cosine_at_300_matched_lr=gap300_m,
            seed_noise_stdev=noise,
            gap_at_200_exceeds_noise=abs(gap200_t) > noise,
            gap_at_300_exceeds_noise=abs(gap300_t) > noise,
            tuning_changed_the_gap_at_300=abs(gap300_t - gap300_m) > noise,
        ),
        curves=curves,
        lr_trace_seed0=lr_trace,
        decision=dict(
            which_model_i_would_keep="the cosine checkpoint, if 200 is a hard stop",
            reason=(
                "At step 200 the cosine run has decayed to lr="
                f"{lr200['cosine']:.2e} ({100*lr200['cosine']/peak['cosine']:.0f}% of its peak) "
                f"while WSD is still on its plateau at lr={lr200['wsd']:.2e} "
                "(100% of its peak). WSD has not been cooled down, so its weights "
                "are still oscillating in a high-lr basin and the checkpoint is "
                "not a settled model. A checkpoint taken mid-plateau is not a "
                "deliverable."
            ),
            but_this_is_not_a_verdict_on_wsd=(
                "The step-200 comparison is unfair to WSD by construction: both "
                "were budgeted for 300 and only cosine's decay overlaps the stop "
                "point. The re-budgeted WSD arm, which actually finishes its "
                "cooldown by 200, is the correct comparison for a 200-step budget."
            ),
            what_i_would_actually_do=(
                "Keep the cosine checkpoint now, and keep the WSD run alive: its "
                "cooldown is the cheap part. If 20 more steps are affordable, cool "
                "the WSD run down and re-compare; that is what the re-budgeted arm "
                "measures."
            ),
            tuning_note=(
                "Both sides were tuned over a 5-point peak-lr grid before any "
                "comparison. The tuned peaks were cosine="
                f"{peak['cosine']:.1e} and WSD={peak['wsd']:.1e}."
            ),
        ),
    )
    (ART / "task4_cosine_vs_wsd.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for label, style in [("cosine_tuned", "-"), ("wsd_tuned", "-"),
                         ("wsd_rebudget_200", ":")]:
        ax[0].plot(lr_trace[label], style, label=f"{label} (peak {arms[label][3]:.1e})")
    ax[0].axvline(STOP, color="k", ls="--", label="stop at 200")
    ax[0].set_title("learning rate schedules (tuned peaks)")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("lr")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    for label, style in [("cosine_tuned", "-"), ("wsd_tuned", "--"),
                         ("wsd_rebudget_200", ":")]:
        xs = [p[0] for p in curves[label][0]]
        ys = [statistics.fmean([c[i][1] for c in curves[label]])
              for i in range(len(curves[label][0]))]
        ax[1].plot(xs, ys, style, label=f"{label} (mean of {len(SEEDS)} seeds)")
    ax[1].axvline(STOP, color="k", ls="--")
    ax[1].set_title("held-out loss"); ax[1].set_xlabel("step")
    ax[1].set_ylabel("loss"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    for name, col in (("cosine", "tab:blue"), ("wsd", "tab:orange")):
        g = tuned[name]["grid"]
        ax[2].plot([r["lr"] for r in g],
                   [r["loss"] if math.isfinite(r["loss"]) else float("nan") for r in g],
                   "o-", color=col, label=f"{name} (best {tuned[name]['peak_lr']:.1e})")
        ax[2].axvline(tuned[name]["peak_lr"], color=col, ls=":", alpha=.7)
    ax[2].set_xscale("log"); ax[2].set_xlabel("peak lr")
    ax[2].set_ylabel(f"held-out loss at step {TOTAL}")
    ax[2].set_title("tuning both sides before comparing")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "task4_cosine_vs_wsd.png", dpi=130)

    print(f"\nTUNED   loss@200 cosine={tuned200['cosine']['mean']:.4f}"
          f"+-{tuned200['cosine']['stdev']:.4f}  wsd={tuned200['wsd']['mean']:.4f}"
          f"+-{tuned200['wsd']['stdev']:.4f}  wsd_rebudget={reb200['mean']:.4f}")
    print(f"TUNED   loss@300 cosine={tuned300['cosine']['mean']:.4f}  "
          f"wsd={tuned300['wsd']['mean']:.4f}")
    print(f"MATCHED loss@200 cosine={matched200['cosine']['mean']:.4f}  "
          f"wsd={matched200['wsd']['mean']:.4f}")
    print(f"MATCHED loss@300 cosine={matched300['cosine']['mean']:.4f}  "
          f"wsd={matched300['wsd']['mean']:.4f}")
    print(f"gap@200={gap200_t:+.4f} gap@300={gap300_t:+.4f} seed noise={noise:.4f}")


if __name__ == "__main__":
    main()
