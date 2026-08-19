"""Synthetic smoke test for NHC: shapes, collate, backward, hazard curve."""
import numpy as np
import torch
import torch.nn.functional as F

from aurora_cf.config import AURORACFConfig
from aurora_cf.data_hazard import (HazardIndex, HazardDataset,
                                   hazard_collate, N_FEAT)
from aurora_cf.model_hazard import NHCModel

rng = np.random.default_rng(0)
NE, NR, T = 40, 5, 30

# synthetic quadruples with a PERIODIC fact injected
quads = []
for _ in range(1500):
    quads.append([rng.integers(NE), rng.integers(NR),
                  rng.integers(NE), rng.integers(T)])
for t in range(0, T, 4):                      # (0,0,1) recurs every 4 steps
    quads.append([0, 0, 1, t])
quads = np.array(quads, dtype=np.int32)

cfg = AURORACFConfig(dataset="WIKI", embed_dim=32, hazard_dim=16,
                     max_support=32, dropout=0.1, use_inverse=True)

idx = HazardIndex(quads, step=1)
ds = HazardDataset(quads[:400], idx, NE, cfg, use_inverse=True,
                   num_relations=NR, desc="smoke")

batch = hazard_collate([ds[i] for i in range(16)])
subs, rels, objs, times, sup_ids, sup_feat, sup_mask = batch
print("sup_ids", tuple(sup_ids.shape), "sup_feat", tuple(sup_feat.shape),
      "F =", N_FEAT)
assert sup_feat.shape[-1] == N_FEAT

model = NHCModel(NE, NR, cfg)
logits, q = model(subs, rels, sup_ids, sup_feat, sup_mask)
print("logits", tuple(logits.shape), "expected", (16, NE))
assert logits.shape == (16, NE)
assert torch.isfinite(logits).all(), "non-finite logits"

loss = F.cross_entropy(logits, objs)
loss.backward()
gnorm = sum(p.grad.norm().item() for p in model.parameters()
            if p.grad is not None)
print(f"loss={loss.item():.4f}  grad_norm_sum={gnorm:.4f}")
assert gnorm > 0, "no gradient reached the parameters"

# hazard branch must receive gradient (this is the whole contribution)
hz_grad = sum(p.grad.abs().sum().item()
              for p in model.hazard_net.parameters() if p.grad is not None)
ph_grad = model.phase_width.grad.abs().sum().item()
print(f"hazard_net grad={hz_grad:.4f}   phase_width grad={ph_grad:.6f}")
assert hz_grad > 0, "hazard net got no gradient"

# non-monotonicity capability check
phases = np.linspace(0.05, 6.0, 40)
c = model.hazard_curve(0, 1, phases, torch.device("cpu")).numpy()
d = np.diff(c)
print(f"hazard curve sign changes = {int((np.diff(np.sign(d)) != 0).sum())} "
      f"(exp-decay baseline would be 0)")

# support recall sanity
print(f"support_recall = {ds.support_recall:.3f}")
print("\nALL SMOKE TESTS PASSED")
