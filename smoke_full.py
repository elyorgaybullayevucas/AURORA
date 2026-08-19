"""Synthetic smoke test for NHC-Full: shapes, gradients, ablations."""
import numpy as np
import torch
import torch.nn.functional as F

from aurora_cf.config import AURORACFConfig
from aurora_cf.data_full import (FullIndex, FullDataset, full_collate, N_FEAT)
from aurora_cf.model_nhc_full import NHCFullModel

rng = np.random.default_rng(0)
NE, NR, T = 60, 6, 40

quads = [[rng.integers(NE), rng.integers(NR), rng.integers(NE),
          rng.integers(T)] for _ in range(2500)]
for t in range(0, T, 4):                    # periodic fact
    quads.append([0, 0, 1, t])
quads = np.array(quads, dtype=np.int32)

base = dict(dataset="WIKI", embed_dim=32, hazard_dim=16, hist_len=6,
            k_neighbors=8, max_support=48, rel_topk=8, n_heads=4,
            conv_channels=16, dropout=0.1, use_inverse=True, num_workers=0)

cfg = AURORACFConfig(**base)
idx = FullIndex(quads, NR, step=1, use_inverse=True, rel_topk=cfg.rel_topk)
ds = FullDataset(quads[:300], idx, NE, cfg, use_inverse=True,
                 num_relations=NR, desc="smoke")

print(f"support_recall={ds.support_recall:.3f}  avg|S|={ds.avg_support:.1f}")
assert ds.avg_support > 5, "support set still degenerate"

batch = full_collate([ds[i] for i in range(12)])
(subs, rels, objs, times, ne, nr, nm, sup_ids, sup_feat, sup_mask) = batch
print("ne", tuple(ne.shape), "sup_feat", tuple(sup_feat.shape), "F", N_FEAT)
assert sup_feat.shape[-1] == N_FEAT

for variant in ["full", "hazard_off", "phase_off"]:
    c = AURORACFConfig(**base, **{variant: True} if variant != "full" else {})
    m = NHCFullModel(NE, NR, c)
    logits, h = m(subs, rels, ne, nr, nm, sup_ids, sup_feat, sup_mask)
    assert logits.shape == (12, NE), logits.shape
    assert torch.isfinite(logits).all(), f"{variant}: non-finite logits"
    loss = F.cross_entropy(logits, objs)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters()
            if p.grad is not None)
    struct_g = sum(p.grad.abs().sum().item()
                   for p in m.decoder.parameters() if p.grad is not None)
    hz_g = sum(p.grad.abs().sum().item()
               for p in m.hazard_net.parameters() if p.grad is not None)
    print(f"  {variant:<12} loss={loss.item():.4f}  grad={g:.2f}  "
          f"decoder={struct_g:.3f}  hazard={hz_g:.3f}")
    assert struct_g > 0, f"{variant}: structural branch got no gradient"
    if variant == "full":
        assert hz_g > 0, "hazard branch got no gradient"
    if variant == "hazard_off":
        assert hz_g == 0, "hazard branch should be inert when off"

# hazard must start at exactly zero so epoch 1 == structural baseline
m = NHCFullModel(NE, NR, AURORACFConfig(**base))
m.eval()
with torch.no_grad():
    l_full, _ = m(subs, rels, ne, nr, nm, sup_ids, sup_feat, sup_mask)
    m.hazard_off = True
    l_str, _ = m(subs, rels, ne, nr, nm, sup_ids, sup_feat, sup_mask)
d = (l_full - l_str).abs().max().item()
print(f"hazard init offset = {d:.2e} (must be 0)")
assert d < 1e-5

# non-monotone capability
m2 = NHCFullModel(NE, NR, AURORACFConfig(**base))
for p in m2.hazard_net[-1].parameters():
    torch.nn.init.normal_(p, std=0.5)
c = m2.hazard_curve(0, 1, np.linspace(0.05, 6.0, 40),
                    torch.device("cpu")).numpy()
sc = int((np.diff(np.sign(np.diff(c))) != 0).sum())
print(f"hazard curve sign changes = {sc}  (exp-decay baseline = 0)")

print("\nALL SMOKE TESTS PASSED")
