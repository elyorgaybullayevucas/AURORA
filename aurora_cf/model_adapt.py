"""
AURORA-ADAPT: Adaptive Copy-Neural TKG Forecasting.

Core innovation:
    gate = f(copy_confidence)   ← STABLE: based on fixed pre-computed copy scores
    score = gate * copy_logit + (1 - gate) * neural_logit

When copy scores are concentrated (one entity clearly dominates) → gate → 1 → copy wins.
When copy scores are diffuse (many candidates equally likely)    → gate → 0 → neural wins.

This automatically handles:
  YAGO/WIKI  (recurrence ~85-90%): copy confident → copy dominates → high accuracy
  ICEWS18/GDELT (recurrence ~50-60%): copy uncertain → neural dominates → learns patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurora_cf.config import AURORACFConfig


class AURORAAdaptModel(nn.Module):

    def __init__(self, num_entities: int, num_relations: int, cfg: AURORACFConfig):
        super().__init__()
        d  = cfg.embed_dim
        R  = num_relations * 2
        self.num_entities = num_entities
        self.cfg = cfg

        # ── Embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── History encoder: masked-average neighbors → GRU ──────────────────
        self.snap_norm = nn.LayerNorm(d)
        self.gru = nn.GRU(
            input_size=d, hidden_size=d,
            num_layers=cfg.gru_layers, batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )

        # ── Query projection ──────────────────────────────────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # ── Per-relation copy blend ───────────────────────────────────────────
        self.rel_copy_mix = nn.Embedding(R, 1)
        nn.init.zeros_(self.rel_copy_mix.weight)

        # ── Adaptive gate network ─────────────────────────────────────────────
        # Input: [log1p(copy_max), copy_conf] — both from FIXED copy scores.
        # Gate is stable during training because inputs never change.
        self.gate_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Init: gate outputs 0 → sigmoid(0) = 0.5, equal mix at start
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    # ── History encoder ───────────────────────────────────────────────────────

    def encode_history(self, subs, ne, nm):
        B, H, K = ne.shape
        ne_c    = ne.clamp(0, self.num_entities - 1)
        h_neigh = self.ent_emb(ne_c)                         # (B,H,K,d)
        mask_f  = nm.float().unsqueeze(-1)                   # (B,H,K,1)
        counts  = mask_f.sum(2).clamp(min=1.0)               # (B,H,1)
        h_snap  = (h_neigh * mask_f).sum(2) / counts         # (B,H,d)
        h_snap  = self.snap_norm(h_snap)

        h_s_exp = self.ent_emb(subs).unsqueeze(1).expand_as(h_snap)
        no_nb   = (counts.squeeze(-1) == 0).float().unsqueeze(-1)
        h_snap  = h_snap * (1 - no_nb) + h_s_exp * no_nb

        _, h_last = self.gru(h_snap)
        return h_last[-1]                                     # (B, d)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, subs, rels, ne, nr, nm, rel_copy, ent_copy):
        N = self.num_entities

        def _pad(x):
            if x.shape[1] < N:
                return F.pad(x, (0, N - x.shape[1]))
            return x[:, :N]

        rel_copy = _pad(rel_copy)
        ent_copy = _pad(ent_copy)

        # 1. Per-relation blended copy score (fixed precomputed signal)
        w          = torch.sigmoid(self.rel_copy_mix(rels))          # (B,1)
        copy_score = w * rel_copy + (1 - w) * ent_copy               # (B,N)
        copy_logit = torch.log1p(copy_score)                         # (B,N)

        # 2. Neural logit
        h_hist = self.encode_history(subs, ne, nm)
        h_s    = self.ent_emb(subs)
        h_r    = self.rel_emb(rels)
        query  = self.query_proj(torch.cat([h_s, h_r, h_hist], -1)) # (B,d)
        neural_logit = query @ self.ent_emb.weight.T                 # (B,N)

        # 3. Copy confidence from FIXED scores — gate is stable during training
        with torch.no_grad():
            c_max  = copy_score.max(dim=-1, keepdim=True)[0]         # (B,1)
            c_sum  = copy_score.sum(dim=-1, keepdim=True).clamp(1e-8)
            c_conf = c_max / c_sum                                    # (B,1) concentration
            gate_in = torch.cat([torch.log1p(c_max), c_conf], dim=-1)# (B,2)

        gate  = torch.sigmoid(self.gate_net(gate_in))                # (B,1)

        # 4. Adaptive combination
        logits = gate * copy_logit + (1 - gate) * neural_logit       # (B,N)
        return logits, query

    def get_query(self, subs, rels, ne, nm):
        h_hist = self.encode_history(subs, ne, nm)
        return self.query_proj(
            torch.cat([self.ent_emb(subs), self.rel_emb(rels), h_hist], -1))
