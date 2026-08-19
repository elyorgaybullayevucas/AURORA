"""KAIROS smoke test on synthetic data with a planted periodic pattern."""
import numpy as np, torch, torch.nn.functional as F
from kairos.config import KairosConfig
from kairos.data import KairosData, identity_collate, N_FEAT
from kairos.model import KAIROS
from kairos.diagnostics import analyse
from aurora_cf.tkg_index import TKGIndex
import os, tempfile

# ── build a tiny dataset on disk with a PERIODIC and a REFRACTORY pattern ────
rng = np.random.default_rng(0)
NE, NR, T = 60, 5, 60
rows = []
for _ in range(3000):
    rows.append((rng.integers(NE), rng.integers(NR), rng.integers(NE),
                 rng.integers(T)))
for t in range(0, T, 6):                 # periodic: fires every 6 steps
    rows.append((0, 0, 1, t))
for t in range(0, T, 6):                 # distractor: fires often, recently
    for d in (1, 2):
        if t + d < T:
            rows.append((0, 0, 2, t + d))
q = np.array(rows, dtype=np.int64)

d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "SYN"), exist_ok=True)
n = len(q)
for nm, sl in (("train", q[:int(n*.8)]), ("valid", q[int(n*.8):int(n*.9)]),
               ("test", q[int(n*.9):])):
    np.savetxt(os.path.join(d, "SYN", nm + ".txt"), sl, fmt="%d", delimiter="\t")

# ── diagnostics ─────────────────────────────────────────────────────────────
idx = TKGIndex(q, NE, NR, step=1, use_inverse=True, rel_topk=8)
print("diagnostics on synthetic test split:")
st = analyse(idx, q[int(n*.9):], 2000, max_support=64)
assert 0.0 <= st["monotone_blocked_of_recurrent"] <= 1.0

# ── data + model ────────────────────────────────────────────────────────────
cfg = KairosConfig(dataset="SYN", data_dir=d, embed_dim=32, hazard_dim=16,
                   gcn_layers=2, conv_channels=8, hist_len=4, max_support=64,
                   rel_topk=8, num_workers=0, query_chunk=4096, dropout=0.1)
import kairos.config as kc
kc.DATASETS["SYN"] = dict(kc.DATASETS["YAGO"])
data = KairosData(cfg)
item = data.train_set[len(data.train_set)//2]
print(f"timestamp={item['t']}  queries={item['subs'].numel()}  "
      f"sup{tuple(item['sup_feat'].shape)}  history={len(item['hist'])}")
assert item["sup_feat"].shape[-1] == N_FEAT

for kw in [{}, {"rec_off": True}, {"struct_off": True}, {"phase_off": True}]:
    c = KairosConfig(**{**vars(cfg), **kw})
    m = KAIROS(NE, NR, c)
    E = m.evolve(item["hist"])
    assert E.shape == (NE, c.embed_dim) and torch.isfinite(E).all()
    lg = m(E, item["subs"], item["rels"], item["sup_ids"],
           item["sup_feat"], item["sup_mask"])
    assert lg.shape == (item["subs"].numel(), NE), lg.shape
    assert torch.isfinite(lg).all(), "non-finite logits"
    F.cross_entropy(lg, item["objs"]).backward()
    ev = sum(p.grad.abs().sum().item() for p in m.evolver.parameters()
             if p.grad is not None)
    tr = sum(p.grad.abs().sum().item() for p in m.trunk.parameters()
             if p.grad is not None)
    tag = list(kw)[0] if kw else "full"
    print(f"  {tag:<12} evolver_grad={ev:10.3f}  trunk_grad={tr:10.3f}")
    if kw.get("rec_off"):
        assert tr == 0
    else:
        assert tr > 0, f"{tag}: recurrence trunk frozen"
    if kw.get("struct_off"):
        assert ev == 0
    else:
        assert ev > 0, f"{tag}: evolver frozen"

# ── kernel can be non-monotone ──────────────────────────────────────────────
m = KAIROS(NE, NR, cfg)
torch.nn.init.normal_(m.w_head.weight, std=0.8)
torch.nn.init.normal_(m.w_head.bias, std=0.8)
ph = np.linspace(0, 10, 80)
c = m.kernel(0, 1, ph, torch.device("cpu")).numpy()
sc = int((np.diff(np.sign(np.diff(c))) != 0).sum())
print(f"kernel sign changes={sc} (exp decay = 0), peak phase={ph[c.argmax()]:.2f}")
assert sc >= 1

# ── blocked-mask matches a direct reference ─────────────────────────────────
from train_kairos import _blocked_mask
sf, sm, si, ob = (item["sup_feat"], item["sup_mask"], item["sup_ids"],
                  item["objs"])
mask = _blocked_mask(sf, sm, si, ob)
ref = []
for i in range(len(ob)):
    cnt = np.expm1(sf[i, :, 0].numpy()); dt = np.expm1(sf[i, :, 1].numpy())
    has = (sf[i, :, 7].numpy() > 0) & sm[i].numpy()
    pos = np.flatnonzero((si[i].numpy() == int(ob[i])) & has)
    if len(pos) == 0:
        ref.append(False); continue
    p = pos[0]
    other = has.copy(); other[si[i].numpy() == int(ob[i])] = False
    ref.append(bool(((dt[other] <= dt[p]) & (cnt[other] >= cnt[p])).any()))
assert mask.tolist() == ref, "blocked mask mismatch"
print("blocked-mask matches reference OK")

# ── tie-aware ranking ───────────────────────────────────────────────────────
from train_kairos import ranks_of
sc = torch.tensor([[5., 3., 3., 3., 1.],      # target 3.0: 1 better, 2 tied
                   [9., 8., 7., 6., 5.],      # target 9.0: strict winner
                   [0., 0., 0., 0., 0.]])     # all tied
tg = torch.tensor([[3.], [9.], [0.]])
r = ranks_of(sc, tg)
expect = torch.tensor([1 + 1 + 2 / 2, 1.0, 1 + 0 + 4 / 2])
assert torch.allclose(r, expect), (r, expect)
opt = (sc > tg).sum(1) + 1
print(f"tie-aware ranks {r.tolist()}  vs optimistic {opt.tolist()}")
assert opt[2].item() == 1 and r[2].item() == 3.0,     "optimistic ranking must differ on the all-tied row"
print("tie-aware ranking OK")
print("\nALL SMOKE TESTS PASSED")
