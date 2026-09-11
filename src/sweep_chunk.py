"""Fill task-5 sweep cells for one width, resuming from the checkpoint.

Why this exists: the sandbox kills background jobs between turns, and the
width-1024 grid costs more wall-clock than a single foreground command is
allowed. So the sweep is driven in chunks. Each cell is checkpointed the moment
it is measured, so any number of interruptions is survivable and no cell is ever
recomputed.

Usage: python3 sweep_chunk.py <width> [max_cells] [budget_seconds]
"""
import json
import math
import pathlib
import statistics
import sys
import time

from task5_lr_width_sweep import ART, LRS, SEEDS, train_one

CKPT = ART / "task5_partial.json"


def main():
    width = int(sys.argv[1])
    max_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0

    done = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    t0 = time.time()
    filled = 0
    for lr in LRS:
        if filled >= max_cells or time.time() - t0 > budget:
            break
        key = f"{width}|{lr:.6g}"
        if key in done:
            continue
        c0 = time.time()
        vals = [train_one(width, lr, s)[0] for s in SEEDS]
        finite = [v for v in vals if math.isfinite(v)]
        done[key] = dict(
            values=vals,
            mean=statistics.fmean(finite) if finite else float("nan"),
            stdev=statistics.stdev(finite) if len(finite) > 1 else 0.0,
            diverged=len(finite) == 0,
        )
        CKPT.write_text(json.dumps(done))
        filled += 1
        print(f"width {width:5d} lr {lr:.2e} -> "
              f"{done[key]['mean']:.4f}   ({time.time()-c0:.0f}s)", flush=True)

    remaining = [lr for lr in LRS if f"{width}|{lr:.6g}" not in done]
    print(f"filled {filled} cells this chunk; {len(remaining)} left at width {width}; "
          f"{len(done)}/{len(LRS)*3} total")


if __name__ == "__main__":
    main()
