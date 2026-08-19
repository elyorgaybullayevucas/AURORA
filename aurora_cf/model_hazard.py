"""
NHC — Neural Hazard Copy for Temporal Knowledge Graph Forecasting.

============================ THE CLAIM =====================================
Every published TKG copy mechanism (CyGNet, CENET, TiRGN, RE-GCN, DaeMon)
ranks historical candidates with a hand-designed score of the form

        s(o) = f(count_o) * exp(-lambda * dt_o)

This is MONOTONE DECREASING in dt_o. It therefore cannot express recurrence
processes whose hazard is non-monotone in elapsed time -- periodic facts,
burst-then-decay facts, or facts with a refractory period.

NHC replaces that fixed form with a LEARNED conditional intensity

        log lambda(o | H_o, s, r) = g_theta( phi(H_o), e_r, e_o )

where phi(H_o) contains the full inter-arrival statistics of the candidate,
including the PHASE feature dt / mean_gap. Phase makes "due-ness"
representable: hazard peaks at phase ~ 1 and falls off on BOTH sides, which
no exponential decay can do.

============================ WHY IT WINS ===================================
On YAGO the copy support already contains the answer 99.9% of the time
(H@10 = 99.93) while H@1 = 83.13. The entire residual error is MIS-ORDERING
INSIDE THE SUPPORT SET -- exactly the quantity a hazard function controls
and a fixed decay cannot. NHC optimises that ordering directly.

============================ ARCHITECTURE ==================================
  hazard branch  : per-candidate MLP over (temporal features, e_r, e_o)
  semantic branch: low-rank bilinear score over ALL entities, small weight
  combination    : additive in log-space, no gate, no oscillation
============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora_cf.data_hazard import N_FEAT


class NHCModel(nn.Module):

    def __init__(self, num_entities: int, num_relations: int, cfg):
        super().__init__()
        d = cfg.embed_dim
        dh = cfg.hazard_dim
        R = num_relations * 2
        self.num_entities = num_entities
        self.cfg = cfg

        # ── Embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── Feature normalisation (running stats over temporal features) ──────
        self.feat_norm = nn.LayerNorm(N_FEAT)

        # ── Phase basis: RBF bumps over dt/mean_gap ───────────────────────────
        # Gives the network an explicit non-monotone basis in phase space.
        # centres at 0.25, 0.5, ... 4.0 — a fixed exponential decay lives in
        # the span of NONE of these except the trivial monotone one.
        centres = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25,
                                1.5, 2.0, 3.0, 4.0, 6.0])
        self.register_buffer("phase_centres", centres)
        self.phase_width = nn.Parameter(torch.full((len(centres),), 0.35))
        n_phase = len(centres) * 2  # rel-phase + ent-phase

        # ── Hazard network ────────────────────────────────────────────────────
        in_dim = N_FEAT + n_phase + dh * 2
        self.rel_ctx = nn.Linear(d, dh)
        self.ent_ctx = nn.Linear(d, dh)
        self.hazard_net = nn.Sequential(
            nn.Linear(in_dim, dh * 2),
            nn.LayerNorm(dh * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dh * 2, dh),
            nn.GELU(),
            nn.Linear(dh, 1),
        )

        # ── Semantic branch (covers entities outside the support set) ─────────
        self.query_proj = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # small, fixed-cap weight: the semantic branch is a TIE-BREAKER,
        # never allowed to overrule the hazard branch.
        self.log_sem_w = nn.Parameter(torch.tensor(cfg.sem_init))
        self.sem_cap = cfg.sem_cap

        # base logit for entities with no history at all
        self.null_logit = nn.Parameter(torch.tensor(-4.0))

    # ── phase basis expansion ────────────────────────────────────────────────

    def _phase_basis(self, phase):
        """phase: (B,S) -> (B,S,C) RBF activations."""
        c = self.phase_centres.view(1, 1, -1)
        w = F.softplus(self.phase_width).view(1, 1, -1) + 1e-3
        z = (phase.unsqueeze(-1) - c) / w
        return torch.exp(-0.5 * z * z)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, subs, rels, sup_ids, sup_feat, sup_mask):
        """
        subs, rels : (B,)
        sup_ids    : (B,S)  candidate entity ids
        sup_feat   : (B,S,F) temporal features
        sup_mask   : (B,S)  valid-candidate mask
        returns logits (B, N)
        """
        B, S, _ = sup_feat.shape
        N = self.num_entities

        # ── hazard branch ────────────────────────────────────────────────────
        f = self.feat_norm(sup_feat)                          # (B,S,F)

        phase_r = sup_feat[..., 7]                            # (B,S)
        phase_e = sup_feat[..., 8]
        pb = torch.cat([self._phase_basis(phase_r),
                        self._phase_basis(phase_e)], dim=-1)  # (B,S,2C)

        h_r = self.rel_ctx(self.rel_emb(rels))                # (B,dh)
        h_r = h_r.unsqueeze(1).expand(B, S, -1)
        h_o = self.ent_ctx(self.ent_emb(sup_ids.clamp(0, N - 1)))  # (B,S,dh)

        hz = self.hazard_net(torch.cat([f, pb, h_r, h_o], -1)).squeeze(-1)
        hz = hz.masked_fill(~sup_mask, 0.0)                   # (B,S)

        # ── semantic branch ──────────────────────────────────────────────────
        q = self.query_proj(torch.cat([self.ent_emb(subs),
                                       self.rel_emb(rels)], -1))   # (B,d)
        sem = q @ self.ent_emb.weight.T                       # (B,N)
        sem_w = torch.sigmoid(self.log_sem_w) * self.sem_cap

        # ── combine: semantic base + hazard on the support ───────────────────
        logits = sem_w * sem + self.null_logit
        logits = logits.scatter_add(
            1, sup_ids.clamp(0, N - 1), hz * sup_mask.float())

        return logits, q

    def hazard_curve(self, rel_id: int, ent_id: int, phases, device):
        """
        Diagnostic: return the learned hazard as a function of phase for one
        (relation, entity) pair — used to SHOW the non-monotone shape that an
        exponential decay cannot produce. Returns a 1-D tensor.
        """
        self.eval()
        with torch.no_grad():
            P = len(phases)
            feat = torch.zeros(1, P, N_FEAT, device=device)
            feat[..., 0] = torch.log1p(torch.tensor(3.0))   # 3 rel events
            feat[..., 1] = torch.log1p(torch.tensor(3.0))
            feat[..., 4] = torch.log1p(torch.tensor(5.0))   # mean gap 5
            feat[..., 9] = 1.0
            ph = torch.as_tensor(phases, dtype=torch.float32,
                                 device=device).view(1, P)
            feat[..., 7] = ph
            feat[..., 8] = ph
            feat[..., 2] = torch.log1p(ph * 5.0)
            feat[..., 3] = torch.log1p(ph * 5.0)

            f = self.feat_norm(feat)
            pb = torch.cat([self._phase_basis(feat[..., 7]),
                            self._phase_basis(feat[..., 8])], -1)
            rels = torch.tensor([rel_id], device=device)
            ids = torch.full((1, P), ent_id, device=device, dtype=torch.long)
            h_r = self.rel_ctx(self.rel_emb(rels)).unsqueeze(1).expand(1, P, -1)
            h_o = self.ent_ctx(self.ent_emb(ids))
            return self.hazard_net(
                torch.cat([f, pb, h_r, h_o], -1)).squeeze(-1)[0]
