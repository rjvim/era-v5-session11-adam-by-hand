"""Task 1: Reproduce Adam by hand for one weight and five gradients.

Hand-computed m, v, m_hat, v_hat and the resulting step are checked against
torch.optim.Adam step-for-step. We use Adam (not AdamW) with weight_decay=0 so
the hand recurrence is the exact textbook Adam update:

    m_t     = b1*m_{t-1} + (1-b1)*g_t
    v_t     = b2*v_{t-1} + (1-b2)*g_t^2
    m_hat_t = m_t / (1 - b1^t)
    v_hat_t = v_t / (1 - b2^t)
    w_t     = w_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

NOTE ON eps PLACEMENT: PyTorch computes
    denom = sqrt(v_t)/sqrt(bias_correction2) + eps
which is algebraically sqrt(v_hat) + eps, i.e. eps sits OUTSIDE the bias
correction on the sqrt. Writing it as sqrt(v_hat + eps) instead is a different
optimizer and disagrees in the low-order digits. We match PyTorch's form.

Everything runs in float64 so the comparison is not limited by fp32 noise.
"""
import json
import math
import pathlib

import torch

ART = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(exist_ok=True)

W0 = 0.3000000000
GRADS = [0.1000000000, -0.2000000000, 0.0500000000, 0.4000000000, -0.1000000000]
LR = 0.1000000000
B1, B2, EPS = 0.9, 0.999, 1e-8


def hand_adam(w0, grads, lr=LR, b1=B1, b2=B2, eps=EPS, bias_correction=True):
    """Pure-Python Adam. No torch, no numpy. This is the 'by hand' arm."""
    w, m, v = w0, 0.0, 0.0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        if bias_correction:
            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)
        else:
            m_hat, v_hat = m, v
        step = lr * m_hat / (math.sqrt(v_hat) + eps)
        w = w - step
        rows.append(
            dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w_after=w)
        )
    return rows


def torch_adam(w0, grads, lr=LR, b1=B1, b2=B2, eps=EPS):
    """torch.optim.Adam, float64, weight_decay=0, no amsgrad, no fused path."""
    p = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
    rows = []
    for t, g in enumerate(grads, start=1):
        p.grad = torch.tensor([g], dtype=torch.float64)
        before = p.item()
        opt.step()
        st = opt.state[p]
        rows.append(
            dict(
                t=t,
                g=g,
                m=st["exp_avg"].item(),
                v=st["exp_avg_sq"].item(),
                m_hat=st["exp_avg"].item() / (1.0 - b1**t),
                v_hat=st["exp_avg_sq"].item() / (1.0 - b2**t),
                step=before - p.item(),
                w_after=p.item(),
            )
        )
    return rows


def main():
    hand = hand_adam(W0, GRADS)
    ref = torch_adam(W0, GRADS)

    fields = ["m", "v", "m_hat", "v_hat", "step", "w_after"]
    comparison, worst = [], {f: 0.0 for f in fields}
    for h, r in zip(hand, ref):
        row = dict(t=h["t"], g=h["g"])
        for f in fields:
            d = abs(h[f] - r[f])
            row[f"hand_{f}"] = h[f]
            row[f"torch_{f}"] = r[f]
            row[f"absdiff_{f}"] = d
            worst[f] = max(worst[f], d)
        # decimal places of agreement for the quantity the assignment cares about
        row["step_decimals_agreeing"] = (
            18
            if row["absdiff_step"] == 0.0
            else int(math.floor(-math.log10(row["absdiff_step"])))
        )
        comparison.append(row)

    worst_any = max(worst.values())
    min_dec = min(r["step_decimals_agreeing"] for r in comparison)

    out = dict(
        setup=dict(
            w0=W0, grads=GRADS, lr=LR, beta1=B1, beta2=B2, eps=EPS,
            dtype="float64", optimizer="torch.optim.Adam", weight_decay=0.0,
            eps_placement="sqrt(v_hat) + eps  (outside bias correction, matches PyTorch)",
        ),
        rows=comparison,
        worst_abs_diff_per_field=worst,
        worst_abs_diff_any_field=worst_any,
        min_step_decimals_agreeing=min_dec,
        final_w_hand=hand[-1]["w_after"],
        final_w_torch=ref[-1]["w_after"],
        verdict="agree to >=12 decimal places" if worst_any < 1e-12 else "MISMATCH",
    )
    (ART / "task1_adam_by_hand.json").write_text(json.dumps(out, indent=2))

    print(f"{'t':>2} {'g':>7} {'m':>13} {'v':>13} {'m_hat':>13} {'v_hat':>13} {'step':>13} {'|d step|':>10}")
    for r in comparison:
        print(
            f"{r['t']:>2} {r['g']:>7.3f} {r['hand_m']:>13.9f} {r['hand_v']:>13.9f} "
            f"{r['hand_m_hat']:>13.9f} {r['hand_v_hat']:>13.9f} {r['hand_step']:>13.9f} "
            f"{r['absdiff_step']:>10.2e}"
        )
    print(f"\nworst abs diff, any field, any step : {worst_any:.3e}")
    print(f"step agrees to at least             : {min_dec} decimal places")
    print(f"final w  hand={hand[-1]['w_after']:.15f}  torch={ref[-1]['w_after']:.15f}")


if __name__ == "__main__":
    main()
