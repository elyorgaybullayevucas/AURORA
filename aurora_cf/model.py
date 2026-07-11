"""
AURORA-CF model: Copy-First architecture.

Core formula:
    score(o) = copy_logit(o) + neural_scale * neural_logit(o)

copy_logit  — strong, pre-computed historical signal (log1p scale)
neural_logit — learned residual correction (inner product)
neural_scale — small scalar, grows slowly during training

This design prevents the neural part from ever "taking over" the copy signal
(the main failure mode of the original AURORA).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurora_cf.config import AURORACFConfig


class AURORACFModel(nn.Module):

    def __init__(self, num_entities: int, num_relations: int, cfg: AURORACFConfig):
        super().__init__()
        d  = cfg.embed_dim
        R  = num_relations * 2  # +inverse relations
        self.cfg = cfg
        self.num_entities = num_entities

        # ── Embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── History encoder: GRU over H snapshot averages ─────────────────────
        # Each snapshot: average of K neighbor embeddings → d-dim vector
        self.snap_norm = nn.LayerNorm(d)
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )

        # ── Query projection ──────────────────────────────────────────────────
        # Input: [h_s || h_r || h_hist]  →  query vector
        self.query_proj = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # ── Per-relation copy mixture ─────────────────────────────────────────
        # w_rel[r] = sigmoid(param) → weight on rel_copy vs ent_copy
        self.rel_copy_mix = nn.Embedding(R, 1)
        nn.init.zeros_(self.rel_copy_mix.weight)  # start at 0.5 (sigmoid(0))

        # ── Neural scale ─────────────────────────────────────────────────────
        # neural_scale = clamp(sigmoid(log_scale) * max_scale, max=scale_cap)
        # scale_cap per dataset prevents neural from ever dominating copy.
        # YAGO/WIKI (high recurrence) → cap=0.15 keeps copy primary.
        # ICEWS18/GDELT (lower recurrence) → cap=1.0 allows neural to grow.
        self.log_neural_scale = nn.Parameter(
            torch.tensor(cfg.neural_init_scale))
        self.max_neural_scale = 2.0
        self.neural_scale_cap = cfg.neural_scale_cap

    # ── Encode history ────────────────────────────────────────────────────────

    def encode_history(self, subs: torch.Tensor,
                       ne: torch.Tensor,
                       nm: torch.Tensor) -> torch.Tensor:
        """
        ne: (B, H, K) neighbor entity IDs (long)
        nm: (B, H, K) bool mask (True = valid neighbor)
        Returns: (B, d)
        """
        B, H, K = ne.shape

        # Clamp to valid entity range
        ne_clamped = ne.clamp(0, self.num_entities - 1)

        # Look up embeddings: (B, H, K, d)
        h_neigh = self.ent_emb(ne_clamped)

        # Masked average over K neighbors → (B, H, d)
        mask_f  = nm.float().unsqueeze(-1)                    # (B, H, K, 1)
        counts  = mask_f.sum(dim=2).clamp(min=1.0)           # (B, H, 1)
        h_snap  = (h_neigh * mask_f).sum(dim=2) / counts     # (B, H, d)
        h_snap  = self.snap_norm(h_snap)

        # Also add subject embedding as context for empty snapshots
        h_s_exp = self.ent_emb(subs).unsqueeze(1).expand_as(h_snap)
        no_nbrs = (counts.squeeze(-1) == 0).float().unsqueeze(-1)  # (B, H, 1)
        h_snap  = h_snap * (1 - no_nbrs) + h_s_exp * no_nbrs

        # GRU over snapshot sequence (index 0 = most recent)
        _, h_last = self.gru(h_snap)     # h_last: (layers, B, d)
        return h_last[-1]                # (B, d)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, subs, rels, ne, nr, nm, rel_copy, ent_copy):
        """
        subs, rels: (B,) long
        ne: (B, H, K) long
        nm: (B, H, K) bool
        rel_copy, ent_copy: (B, N) float — pre-computed copy scores
        Returns: logits (B, N)
        """
        N = self.num_entities

        # Pad copy scores if collate produced smaller N
        def _pad(x):
            if x.shape[1] < N:
                return F.pad(x, (0, N - x.shape[1]))
            return x[:, :N]

        rel_copy = _pad(rel_copy)
        ent_copy = _pad(ent_copy)

        # 1. History encoding
        h_hist = self.encode_history(subs, ne, nm)       # (B, d)

        # 2. Entity / relation embeddings
        h_s = self.ent_emb(subs)    # (B, d)
        h_r = self.rel_emb(rels)    # (B, d)

        # 3. Query representation
        query = self.query_proj(
            torch.cat([h_s, h_r, h_hist], dim=-1))       # (B, d)

        # 4. Neural logit: inner product with ALL entity embeddings
        neural_logit = query @ self.ent_emb.weight.T     # (B, N)

        # 5. Copy prior: per-relation blend of rel_copy and ent_copy
        w = torch.sigmoid(self.rel_copy_mix(rels))       # (B, 1)
        copy_score = w * rel_copy + (1 - w) * ent_copy  # (B, N)
        copy_logit = torch.log1p(copy_score)             # log(1 + score) ≥ 0

        # 6. Additive combination — copy dominates, neural corrects
        scale = torch.sigmoid(self.log_neural_scale) * self.max_neural_scale
        scale = scale.clamp(max=self.neural_scale_cap)
        logits = copy_logit + scale * neural_logit       # (B, N)

        return logits, query

    def get_query(self, subs, rels, ne, nm):
        """For InfoNCE loss — returns query repr only."""
        h_hist = self.encode_history(subs, ne, nm)
        h_s    = self.ent_emb(subs)
        h_r    = self.rel_emb(rels)
        return self.query_proj(torch.cat([h_s, h_r, h_hist], dim=-1))
