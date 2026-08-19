"""
HAWK smoke test.

Checks the things that actually went wrong in earlier iterations:
  1. support set is not degenerate (avg|S| >> 1)
  2. index contains inverse edges (inverse queries have real history)
  3. recurrence branch starts small and is inert in the low-lambda limit
  4. every ablation runs and routes gradient to the right branch
  5. superposition can only raise a candidate's score, never lower it
  6. the learned hazard can be non-monotone in phase
  7. filtered ranking matches a naive reference implementation
"""
import numpy as np
import torch
import torch.nn.functional as F

from aurora_cf.hawk_config import HawkConfig
from aurora_cf.tkg_index import TKGIndex
from aurora_cf.data_tpp import TPPDataset, tpp_collate, N_FEAT
from aurora_cf.model_hawk import HAWK

rng = np.random.default_rng(0)
NE, NR, T = 80, 6, 50

q = [[rng.integers(NE), rng.integers(NR), rng.integers(NE), rng.integers(T)]
     for _ in range(4000)]
for t in range(0, T, 5):                       # periodic fact
    q.append([0, 0, 1, t])
quads = np.array(q, dtype=np.int64)

cfg = HawkConfig(dataset="WIKI", embed_dim=64, hazard_dim=32, dt_dim=8,
                 n_heads=4, n_layers=2, gru_layers=1, conv_channels=16,
                 hist_len=5, k_neighbors=8, max_support=48, rel_topk=8,
                 dropout=0.1, num_workers=0)

idx = TKGIndex(quads, NE, NR, step=1, use_inverse=True, rel_topk=cfg.rel_topk)
print(f"index edges={idx.n_edges:,}  (raw={len(quads):,} → inverse added)")
assert idx.n_edges == 2 * len(quads)

ds = TPPDataset(quads[:400], idx, cfg, use_inverse=True, num_relations=NR)

# ── 1 & 2: support quality, forward vs inverse ───────────────────────────────
sizes, hit = [], 0
fwd_hit = inv_hit = fwd_n = inv_n = 0
for i in range(len(ds)):
    s, r, o, t = ds.data[i]
    ids, _ = idx.candidates(s, r, t, cfg.max_support)
    sizes.append(len(ids))
    ok = o in ids
    hit += ok
    if r < NR:
        fwd_hit += ok; fwd_n += 1
    else:
        inv_hit += ok; inv_n += 1
print(f"support: recall={hit/len(ds):.3f}  avg|S|={np.mean(sizes):.1f}")
print(f"         forward recall={fwd_hit/max(fwd_n,1):.3f}  "
      f"inverse recall={inv_hit/max(inv_n,1):.3f}")
assert np.mean(sizes) > 5, "support set degenerate"
# On uniform synthetic data the absolute recall is low (there is no real
# recurrence to find). What must hold is that the two directions behave the
# same: an index missing inverse edges drives inverse recall to ~0.
fr = fwd_hit / max(fwd_n, 1)
ir = inv_hit / max(inv_n, 1)
assert abs(fr - ir) < 0.15 and ir > 0.1, \
    f"forward/inverse asymmetry: {fr:.3f} vs {ir:.3f} — inverse edges missing"

batch = tpp_collate([ds[i] for i in range(16)])
(subs, rels, objs, times, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask) = batch
print(f"ne{tuple(ne.shape)}  sup_feat{tuple(sup_feat.shape)}  F={N_FEAT}")

# ── 3: epoch-1 equals structural-only ────────────────────────────────────────
torch.manual_seed(0); m = HAWK(NE, NR, cfg).eval()
with torch.no_grad():
    full = m(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
    m.hazard_off = True
    stru = m(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
    m.hazard_off = False
d = (full - stru).abs().max().item()
with torch.no_grad():
    m.rec_bias.fill_(-20.0)
    far = m(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
    m.rec_bias.fill_(cfg.rec_bias_init)
d_lim = (far - stru).abs().max().item()
print(f"init |full - structural| = {d:.2e}   "
      f"limit (rec_bias=-20) = {d_lim:.2e}")
# A nonzero initial perturbation is intended: driving it to zero would also
# drive the gradient reaching f_rec, sigmoid(f_rec - f_nov), to zero. What is
# checked is that it is small next to the logit scale, and that the branch is
# exactly inert once the intensity is negligible.
assert d < 0.25, "recurrence branch perturbs the structural model too much"
assert d_lim < 1e-3, "recurrence branch is not inert in the low-intensity limit"

# ── 5: superposition is monotone-increasing in the recurrence term ───────────
with torch.no_grad():
    m.rec_bias.fill_(2.0)
    raised = m(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
    m.rec_bias.fill_(cfg.rec_bias_init)
assert (raised >= stru - 1e-4).all(), "superposition lowered a score"
print("superposition monotonicity ✓")

# ── 4: ablations ─────────────────────────────────────────────────────────────
for kw in [{}, {"hazard_off": True}, {"struct_off": True}, {"phase_off": True}]:
    c = HawkConfig(**{**vars(cfg), **kw})
    mm = HAWK(NE, NR, c)
    lg = mm(subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask)
    assert lg.shape == (16, NE) and torch.isfinite(lg).all()
    F.cross_entropy(lg, objs).backward()
    dec = sum(p.grad.abs().sum().item() for p in mm.decoder.parameters()
              if p.grad is not None)
    haz = sum(p.grad.abs().sum().item() for p in mm.trunk.parameters()
              if p.grad is not None)
    tag = list(kw)[0] if kw else "full"
    print(f"  {tag:<12} decoder_grad={dec:9.3f}  hazard_grad={haz:9.3f}")
    if kw.get("hazard_off"):
        assert haz == 0, "hazard branch should be inert when disabled"
    else:
        # every variant that keeps the recurrence branch must actually train
        # it -- a zero here means a dead gradient path, not a small effect
        assert haz > 0, f"{tag}: recurrence trunk receives no gradient"
    if kw.get("struct_off"):
        assert dec == 0
    else:
        assert dec > 0

# ── 6: non-monotone hazard is reachable ──────────────────────────────────────
m3 = HAWK(NE, NR, cfg)
torch.nn.init.normal_(m3.w_head.weight, std=0.8)
torch.nn.init.normal_(m3.w_head.bias, std=0.8)
ph = np.linspace(0.0, 8.0, 60)
c3 = m3.hazard_curve(0, 1, ph, torch.device("cpu")).numpy()
sign_changes = int((np.diff(np.sign(np.diff(c3))) != 0).sum())
print(f"hazard curve sign changes = {sign_changes} "
      f"(exp(-λΔt) baseline = 0), peak at phase={ph[c3.argmax()]:.2f}")
assert sign_changes >= 1, "hazard is monotone — novelty not realised"

# ── 7: filtered ranking vs naive reference ───────────────────────────────────
torch.manual_seed(1)
logits = torch.randn(16, NE)
naive = []
for i in range(16):
    sc = logits[i].clone()
    o = int(objs[i])
    ans = idx.answers(int(subs[i]), int(rels[i]), int(times[i]))
    for a in ans:
        if a != o:
            sc[a] = float("-inf")
    naive.append(int((sc > sc[o]).sum().item()) + 1)

lg = logits.clone()
rows, cols = [], []
for i in range(16):
    a = idx.answers(int(subs[i]), int(rels[i]), int(times[i]))
    if len(a):
        cols.append(a); rows.append(np.full(len(a), i, dtype=np.int64))
tgt = lg.gather(1, objs.view(-1, 1))
if rows:
    lg.index_put_((torch.from_numpy(np.concatenate(rows)),
                   torch.from_numpy(np.concatenate(cols).astype(np.int64))),
                  torch.full((sum(len(c) for c in cols),), float("-inf")))
lg.scatter_(1, objs.view(-1, 1), tgt)
vec = ((lg > tgt).sum(1) + 1).tolist()
assert vec == naive, f"ranking mismatch\n{vec}\n{naive}"
print("filtered ranking matches reference ✓")

print("\nALL SMOKE TESTS PASSED")
