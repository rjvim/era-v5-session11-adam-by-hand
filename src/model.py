"""Shared tiny decoder-only transformer and a deterministic synthetic task.

Kept small on purpose: the box running this has 1 CPU core, so a large model
would force the sweeps to be too short to mean anything. The design choice is
to spend the compute on MANY runs (sweeps, controls, seeds) rather than one big
run, because every question in this assignment is a comparison.

THE TASK IS SYNTHETIC AND LEARNABLE, NOT RANDOM. Random tokens have no signal,
so loss would sit at ln(V) forever and every lr would look identical, which
would silently invalidate tasks 4 and 5. Instead we use a deterministic
"induction / copy with offset" task: the target is a fixed permutation of the
token k positions back. A transformer can learn this, so loss actually moves and
lr differences show up as real loss differences.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 64
SEQLEN = 64
COPY_OFFSET = 3


def make_batch(batch=8, seqlen=SEQLEN, gen=None, vocab=VOCAB, noise_frac=0.30):
    """x: random tokens. y: a deterministic function of TWO past positions,
    plus irreducible label noise.

    Design history (this matters, and the first version was wrong):
    v1 was a single-position copy, y[t] = f(x[t-3]). A 2-layer transformer
    solves that almost perfectly, so loss collapsed to ~0.02 and EVERY schedule
    and EVERY learning rate looked the same. A saturated task cannot
    discriminate between optimizers, which would have silently invalidated
    tasks 4 and 5. Verified: the v1 run reached 0.0013 loss by step 300.

    v2 (this one) is harder in two deliberate ways:
      1. A fixed fraction of labels is replaced with noise, which puts a
         nonzero floor under the loss. Without a floor, the best lr is just
         "the largest that does not diverge" and the sweep has no interior
         minimum to find.
      2. Vocabulary reduced to 64 so the head does not dominate the parameter
         count at small widths, which would confound the width sweep.

    An intermediate two-lag version was ALSO tried and rejected: it was too
    hard for a 2-layer model in 300 steps (loss pinned at ln(64)=4.16, i.e. no
    learning at all), which is just as uninformative as saturation. Measured:
    two-lag 4.05 vs single-lag 2.18 at step 300, against a ceiling of 4.16.

    y[t] = (x[t-3]*7 + 11) % vocab, then noise_frac of positions randomised.
    Loss floor is about noise_frac * ln(vocab) = 1.25; ceiling ln(64) = 4.16.
    """
    if gen is None:
        gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab, (batch, seqlen), generator=gen)
    a = torch.zeros_like(x)
    a[:, COPY_OFFSET:] = x[:, :-COPY_OFFSET]
    y = (a * 7 + 11) % vocab
    if noise_frac > 0:
        mask = torch.rand(x.shape, generator=gen) < noise_frac
        rnd = torch.randint(0, vocab, x.shape, generator=gen)
        y = torch.where(mask, rnd, y)
    return x, y


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, h):
        B, T, C = h.shape
        x = self.ln1(h)
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        h = h + self.proj(a)
        h = h + self.fc2(F.gelu(self.fc1(self.ln2(h))))
        return h


class TinyLM(nn.Module):
    """Width is the knob task 5 sweeps. Depth and heads held fixed."""

    def __init__(self, d_model=128, n_layer=2, n_head=4, vocab=VOCAB,
                 seqlen=SEQLEN, seed=0, mup_scaling=False, base_width=256):
        super().__init__()
        torch.manual_seed(seed)
        self.d_model = d_model
        self.mup_scaling = mup_scaling
        self.base_width = base_width
        n_head = max(1, min(n_head, d_model // 16))
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seqlen, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)

    def _init(self, m):
        # Standard-parameterization init: std = 0.02, width-independent.
        # Task 5 relies on this being SP, not muP, since the whole point is to
        # show that the optimal lr MOVES with width under SP.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))

    def loss(self, x, y):
        logits = self(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
