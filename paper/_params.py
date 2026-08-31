"""Parameter counts quoted in the paper. Run from the repo root."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kairos.config import KairosConfig
from kairos.model import KAIROS

REC = ("rel_ctx", "ent_ctx", "sub_ctx", "feat_norm", "trunk", "rec_head",
       "mono_trunk", "mono_head", "rec_bias")

for name, ne, nr in [("YAGO", 10623, 10), ("ICEWS18", 23033, 256)]:
    cfg = KairosConfig(dataset=name)
    m = KAIROS(ne, nr, cfg)
    tot = sum(p.numel() for p in m.parameters())
    rec = sum(p.numel() for n, p in m.named_parameters()
              if n.split(".")[0] in REC)
    print(f"{name:<8} entities={ne:<6} total={tot:,}  recurrence={rec:,}")
