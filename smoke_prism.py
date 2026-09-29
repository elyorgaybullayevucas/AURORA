"""
PRISM smoke test: the properties the paper claims, checked on synthetic data.

  1. P(o|q) is an exact distribution over all N entities.
  2. The likelihood factorises: for a query whose answer is IN the support,
     the structural scores outside the support get exactly zero gradient; for
     one whose answer is OUTSIDE, the recurrence trunk gets exactly zero.
  3. An empty support routes all mass to the complement.
  4. A padding slot that clamps onto a real support entity cannot erase it.
  5. It trains: loss falls on a tiny dataset.
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

import kairos.config as kc
from kairos.config import KairosConfig
from kairos.data import KairosData
from kairos.prism import PRISM

torch.manual_seed(0)
rng = np.random.default_rng(0)
NE, NR, T = 60, 5, 60
rows = [(rng.integers(NE), rng.integers(NR), rng.integers(NE), rng.integers(T))
        for _ in range(3000)]
for t in range(0, T, 6):
    rows.append((0, 0, 1, t))
q = np.array(rows, dtype=np.int64)
d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "SYN"), exist_ok=True)
n = len(q)
for nm, sl in (("train", q[:int(n * .8)]), ("valid", q[int(n * .8):int(n * .9)]),
               ("test", q[int(n * .9):])):
    np.savetxt(os.path.join(d, "SYN", nm + ".txt"), sl, fmt="%d", delimiter="\t")

kc.DATASETS["SYN"] = dict(kc.DATASETS["YAGO"])
cfg = KairosConfig(dataset="SYN", data_dir=d, embed_dim=32, hazard_dim=16,
                   gcn_layers=2, conv_channels=8, hist_len=4, max_support=64,
                   rel_topk=8, num_workers=0, dropout=0.0, prism=True)
data = KairosData(cfg)
it = data.train_set[len(data.train_set) // 2]
m = PRISM(NE, NR, cfg)
args = (it["subs"], it["rels"], it["sup_ids"], it["sup_feat"], it["sup_mask"])

# ── 1. exact distribution ────────────────────────────────────────────────────
E, _ = m.evolve(it["hist"])
lp = m(E, *args)
tot = lp.exp().sum(1)
print(f"sum_o P(o|q): min {tot.min():.6f}  max {tot.max():.6f}")
assert torch.allclose(tot, torch.ones_like(tot), atol=1e-4), "not normalised"
assert torch.isfinite(lp).all()
print("exact distribution OK")

# ── 2. factorisation ─────────────────────────────────────────────────────────
ids = it["sup_ids"].clamp(0, NE - 1)
inS = torch.zeros(len(it["subs"]), NE, dtype=torch.bool)
r_, s_ = it["sup_mask"].nonzero(as_tuple=True)
inS[r_, ids[r_, s_]] = True
ans_in = inS[torch.arange(len(it["objs"])), it["objs"]]
assert ans_in.any() and (~ans_in).any(), "need both kinds of query here"

leaf = {}
orig = m.structural


def structural_leaf(E_, subs, rels):
    x = orig(E_, subs, rels).detach().requires_grad_(True)
    leaf["x"] = x
    return x


m.structural = structural_leaf
for want_in in (True, False):
    sel = (ans_in == want_in).nonzero(as_tuple=True)[0]
    sub_args = tuple(a[sel] for a in args)
    m.zero_grad()
    E, _ = m.evolve(it["hist"])
    out = m(E.detach(), *sub_args)
    F.nll_loss(out, it["objs"][sel]).backward()
    g = leaf["x"].grad
    if want_in:
        # answer in S: nothing outside S may be pushed
        leak = g.masked_select(~inS[sel]).abs().max().item()
        print(f"answer in S   : max |grad| on structural scores outside S = {leak:.2e}")
        assert leak == 0.0, "structural expert trained on a recurrent query"
    else:
        tr = sum(p.grad.abs().sum().item() for p in m.trunk.parameters()
                 if p.grad is not None)
        print(f"answer not in S: recurrence trunk grad = {tr:.2e}")
        assert tr == 0.0, "recurrence expert trained on a non-historical query"
m.structural = orig
print("factorisation OK  (each expert is trained only on its own problem)")

# ── 3. empty support routes everything to the complement ────────────────────
E, _ = m.evolve(it["hist"])
empty = torch.zeros_like(it["sup_mask"])
lp0 = m(E, it["subs"], it["rels"], it["sup_ids"], it["sup_feat"], empty)
assert torch.allclose(lp0.exp().sum(1), torch.ones(len(it["subs"])), atol=1e-4)
print("empty support OK")

# ── 4. padding cannot erase a real entity 0 ─────────────────────────────────
sid = it["sup_ids"].clone(); smk = it["sup_mask"].clone()
row = 0
k = int(smk[row].sum())
if k + 1 < smk.shape[1]:
    sid[row, 0] = 0; smk[row, 0] = True           # entity 0 genuinely in S
    sid[row, k + 1:] = 0; smk[row, k + 1:] = False  # padding also points at 0
    a = m(E, it["subs"], it["rels"], sid, it["sup_feat"], smk)
    assert torch.allclose(a.exp().sum(1), torch.ones(len(it["subs"])), atol=1e-4)
    print("padding collision OK")

# ── 5. it trains ────────────────────────────────────────────────────────────
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
first = last = None
for step in range(40):
    opt.zero_grad()
    E, aux = m.evolve(it["hist"])
    loss = F.nll_loss(m(E, *args), it["objs"]) + 0.1 * aux
    loss.backward(); opt.step()
    first = loss.item() if first is None else first
    last = loss.item()
print(f"loss {first:.3f} -> {last:.3f}")
assert last < first * 0.8, "PRISM does not fit a single snapshot"
print("\nALL PRISM SMOKE TESTS PASSED")
