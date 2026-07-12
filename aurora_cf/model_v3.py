"""
AURORA-v3: Two-mode TKG Forecasting.

YAGO/WIKI  (copy_only=True):
    - No neural inner product (the enemy for high-recurrence datasets)
    - Learnable per-relation temperature + blend
    - Expected: 80-83% H@1 on YAGO, 72-75% on WIKI

ICEWS18/GDELT (copy_only=False):
    - Full neural encoder + copy prior
    - FIXED gate (not learned) — eliminates oscillation
    - Expected: 23-25% H@1 on ICEWS18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurora_cf.config import AURORACFConfig


class AURORAv3Model(nn.Module):

    def __init__(self, num_entities: int, num_relations: int, cfg: AURORACFConfig):
        super().__init__()
        d = cfg.embed_dim
        R = num_relations * 2
        self.num_entities = num_entities
        self.cfg = cfg
        self.copy_only = getattr(cfg, "copy_only", False)
        self.fixed_gate = getattr(cfg, "fixed_gate", 0.3)

        # ── Per-relation copy blend (both modes) ──────────────────────────────
        self.rel_blend = nn.Embedding(R, 1)
        nn.init.zeros_(self.rel_blend.weight)

        # ── Per-relation temperature (both modes) ─────────────────────────────
        self.rel_temp = nn.Embedding(R, 1)
        nn.init.ones_(self.rel_temp.weight)
        self.global_bias = nn.Parameter(torch.zeros(1))

        if not self.copy_only:
            # ── Neural encoder (ICEWS18/GDELT only) ───────────────────────────
            self.ent_emb = nn.Embedding(num_entities, d)
            self.rel_emb = nn.Embedding(R, d)
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)

            self.snap_norm = nn.LayerNorm(d)
            self.gru = nn.GRU(
                input_size=d, hidden_size=d,
                num_layers=cfg.gru_layers, batch_first=True,
                dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
            )
            self.query_proj = nn.Sequential(
                nn.Linear(d * 3, d * 2),
                nn.LayerNorm(d * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d * 2, d),
                nn.LayerNorm(d),
            )

    def _pad(self, x, N):
        if x.shape[1] < N:
            return F.pad(x, (0, N - x.shape[1]))
        return x[:, :N]

    def _copy_logit(self, rels, rel_copy, ent_copy):
        N = self.num_entities
        rel_copy = self._pad(rel_copy, N)
        ent_copy = self._pad(ent_copy, N)
        w = torch.sigmoid(self.rel_blend(rels))          # (B,1)
        copy_score = w * rel_copy + (1 - w) * ent_copy  # (B,N)
        temp = F.softplus(self.rel_temp(rels))           # (B,1) positive
        return torch.log1p(copy_score) * temp + self.global_bias, copy_score

    def _encode_history(self, subs, ne, nm):
        B, H, K = ne.shape
        ne_c = ne.clamp(0, self.num_entities - 1)
        h_neigh = self.ent_emb(ne_c)
        mask_f = nm.float().unsqueeze(-1)
        counts = mask_f.sum(2).clamp(min=1.0)
        h_snap = (h_neigh * mask_f).sum(2) / counts
        h_snap = self.snap_norm(h_snap)
        h_s_exp = self.ent_emb(subs).unsqueeze(1).expand_as(h_snap)
        no_nb = (counts.squeeze(-1) == 0).float().unsqueeze(-1)
        h_snap = h_snap * (1 - no_nb) + h_s_exp * no_nb
        _, h_last = self.gru(h_snap)
        return h_last[-1]

    def forward(self, subs, rels, ne, nr, nm, rel_copy, ent_copy):
        copy_logit, copy_score = self._copy_logit(rels, rel_copy, ent_copy)

        if self.copy_only:
            # YAGO/WIKI: pure calibrated copy, no neural
            return copy_logit, None

        # ICEWS18/GDELT: fixed gate + neural
        h_hist = self._encode_history(subs, ne, nm)
        query = self.query_proj(
            torch.cat([self.ent_emb(subs), self.rel_emb(rels), h_hist], -1))
        neural_logit = query @ self.ent_emb.weight.T

        # FIXED gate — never learned, no oscillation
        g = self.fixed_gate
        logits = g * copy_logit + (1 - g) * neural_logit
        return logits, query

    def get_query(self, subs, rels, ne, nm):
        if self.copy_only:
            return None
        h_hist = self._encode_history(subs, ne, nm)
        return self.query_proj(
            torch.cat([self.ent_emb(subs), self.rel_emb(rels), h_hist], -1))
