"""
KAIROS — phase-conditioned recurrence for temporal knowledge graph forecasting.

(kairos: the opportune moment. The model is about WHEN a fact is due.)

================================= POSITION ==================================
Recurrence dominates TKG forecasting. "History Repeats Itself" (IJCAI 2024)
showed that a plain recurrency baseline matches or beats RE-GCN, CyGNet,
TiRGN and L2-TKG on the standard benchmarks. So the interesting question is
not whether to model recurrence, but what the existing recurrence models
cannot express.

Every one of them scores a historical candidate with

        s(o) = f(count_o) * g(dt_o),        g non-increasing in dt

  CyGNet, CENET, TiRGN, RE-GCN, DaeMon : g(dt) = exp(-lambda * dt), lambda fixed
  GAttNHP (2026)                       : g(dt) = exp(-gamma_u * dt), gamma learned

PROPOSITION. Let o* be the answer and o a distractor with dt_o <= dt_{o*} and
count_o >= count_{o*}. Then for every non-decreasing f and non-increasing g,
s(o) >= s(o*). No choice of lambda, no per-entity learned decay rate, and no
reweighting of counts ranks o* strictly first.

Queries of that shape are what we call MONOTONE-BLOCKED. Their rate is a
property of the data and is measured by diagnose.py before any training.
They arise wherever recurrence is phase-structured rather than decaying:
periodic facts, refractory periods, burst-then-die processes.

================================ THE MODEL ==================================
A query is a draw from a superposition of two marked point processes,

    lambda(o) = lambda_struct(o | G_<t, s, r)  +  lambda_rec(o | H_o, s, r)

Superposition adds intensities, so the branches combine through logaddexp,
not by summing logits (summing logits multiplies intensities, which has no
point-process meaning). There is no mixing gate, hence no gate to collapse.

  lambda_struct : entity states evolved through the H preceding snapshots by
                  a composition-based relational GCN with a GRU across
                  snapshots, decoded by ConvTransE against all entities.
                  This is the RE-GCN/TiRGN regime -- states are updated from
                  the whole graph, not from a local neighbourhood.

  lambda_rec    : log lambda_rec(o) = log sum_j w_j(r, o, phi_o) kappa_j(p_o)
                  p_o = dt_o / mean_gap_o                        (PHASE)
                  kappa a fixed RBF basis, w_j >= 0 predicted per candidate.
                  Non-monotone in dt by construction, so monotone-blocked
                  queries become reachable.

EXPRESSIVENESS. Setting w_j to fit exp(-lambda * p) recovers the classical
decay term, so every model above is a special case up to basis resolution.
--phase_off removes the basis and restores a monotone-only scorer, which is
the ablation that isolates the claim.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kairos.data import N_FEAT


# ── structural branch: evolving entity representations ───────────────────────

class CompGCNLayer(nn.Module):
    """Composition-based relational aggregation over one snapshot."""

    def __init__(self, d, dropout):
        super().__init__()
        self.w_msg = nn.Linear(d, d, bias=False)
        self.w_self = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, E, R, src, rel, dst):
        msg = self.w_msg(E[src] + R[rel])                 # (M, d)
        agg = torch.zeros_like(E)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(E.size(0), 1, device=E.device, dtype=E.dtype)
        deg.index_add_(0, dst, torch.ones(len(dst), 1, device=E.device,
                                          dtype=E.dtype))
        agg = agg / deg.clamp(min=1.0)
        return self.norm(self.drop(F.relu(agg + self.w_self(E))))


class Evolver(nn.Module):
    """RGCN inside a snapshot, GRU across snapshots — over ALL entities."""

    def __init__(self, d, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [CompGCNLayer(d, dropout) for _ in range(n_layers)])
        self.cell = nn.GRUCell(d, d)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, E0, R, history):
        """history: list of (src, rel, dst), oldest first."""
        E = E0
        for (src, rel, dst) in history:
            if len(src) == 0:
                continue
            H = E
            for layer in self.layers:
                H = layer(H, R, src, rel, dst)
            E = self.cell(H, E)
            u = torch.sigmoid(self.gate(torch.cat([E, E0], -1)))
            E = u * E + (1 - u) * E0            # anchor to static embeddings
        return F.normalize(E, dim=-1) * math.sqrt(E.size(-1)) * 0.5 + E0


class ConvTransE(nn.Module):
    def __init__(self, d, channels, kernel, dropout):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(2)
        self.conv = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc = nn.Linear(channels * d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.d0 = nn.Dropout(dropout)
        self.d1 = nn.Dropout(dropout)

    def forward(self, h_s, h_r, table, bias):
        B = h_s.size(0)
        x = self.d0(self.bn0(torch.stack([h_s, h_r], 1)))
        x = self.d1(F.relu(self.bn1(self.conv(x))))
        x = F.relu(self.bn2(self.fc(x.reshape(B, -1))))
        return x @ table.T + bias


# ── KAIROS ───────────────────────────────────────────────────────────────────

class KAIROS(nn.Module):

    def __init__(self, num_entities, num_relations, cfg):
        super().__init__()
        d, dh = cfg.embed_dim, cfg.hazard_dim
        R2 = num_relations * 2
        self.N = num_entities
        self.rec_off = cfg.rec_off
        self.struct_off = cfg.struct_off
        self.phase_off = cfg.phase_off

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R2, d)
        self.ent_bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.normal_(self.ent_emb.weight, std=0.02)
        nn.init.normal_(self.rel_emb.weight, std=0.02)

        self.evolver = Evolver(d, cfg.gcn_layers, cfg.dropout)
        self.decoder = ConvTransE(d, cfg.conv_channels, 3, cfg.dropout)

        # ── recurrence kernel ────────────────────────────────────────────────
        centres = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5,
                                2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 13.0])
        self.register_buffer("centres", centres)
        self.log_width = nn.Parameter(
            torch.full((len(centres),), math.log(0.3)))
        self.n_basis = len(centres)
        self.n_ctx = 3                        # phases: (s,r), (s,.), (r,.)

        self.feat_norm = nn.LayerNorm(N_FEAT)
        self.rel_ctx = nn.Linear(d, dh)
        self.ent_ctx = nn.Linear(d, dh)
        self.trunk = nn.Sequential(
            nn.Linear(N_FEAT + 2 * dh, 2 * dh), nn.LayerNorm(2 * dh),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * dh, 2 * dh), nn.LayerNorm(2 * dh), nn.GELU(),
        )
        self.w_head = nn.Linear(2 * dh, self.n_basis * self.n_ctx)
        self.mono_head = nn.Linear(2 * dh, 1)
        # small, NOT zero: a zero weight matrix makes du/dz = 0 and freezes
        # the trunk for the whole run
        nn.init.normal_(self.w_head.weight, std=0.02)
        nn.init.zeros_(self.w_head.bias)
        nn.init.normal_(self.mono_head.weight, std=0.02)
        nn.init.zeros_(self.mono_head.bias)
        self.rec_bias = nn.Parameter(torch.tensor(cfg.rec_bias_init))

    # ── branches ─────────────────────────────────────────────────────────────

    def evolve(self, history):
        if self.struct_off:
            return self.ent_emb.weight
        return self.evolver(self.ent_emb.weight, self.rel_emb.weight, history)

    def structural(self, E, subs, rels):
        if self.struct_off:
            return self.ent_bias.unsqueeze(0).expand(len(subs), -1)
        return self.decoder(E[subs], self.rel_emb(rels), E, self.ent_bias)

    def recurrence(self, rels, sup_ids, sup_feat):
        B, S, _ = sup_feat.shape
        x = torch.cat([
            self.feat_norm(sup_feat),
            self.rel_ctx(self.rel_emb(rels)).unsqueeze(1).expand(B, S, -1),
            self.ent_ctx(self.ent_emb(sup_ids.clamp(0, self.N - 1))),
        ], -1)
        z = self.trunk(x)

        if self.phase_off:
            return self.mono_head(z).squeeze(-1) + self.rec_bias

        w = F.softplus(self.w_head(z)).view(B, S, self.n_ctx, self.n_basis)
        p = torch.stack([sup_feat[..., 5], sup_feat[..., 11],
                         sup_feat[..., 15]], 2)               # (B,S,ctx)
        c = self.centres.view(1, 1, 1, -1)
        wd = self.log_width.exp().view(1, 1, 1, -1).clamp(1e-2, 8.0)
        k = torch.exp(-0.5 * ((p.unsqueeze(-1) - c) / wd) ** 2)
        lam = (w * k).sum(-1).sum(-1)                          # (B,S) >= 0
        return torch.log(lam + 1e-8) + self.rec_bias

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, E, subs, rels, sup_ids, sup_feat, sup_mask):
        f_struct = self.structural(E, subs, rels)
        if self.rec_off:
            return f_struct

        f_rec = self.recurrence(rels, sup_ids, sup_feat)
        f_rec = f_rec.masked_fill(~sup_mask, -1e4)

        ids = sup_ids.clamp(0, self.N - 1)
        base = f_struct.gather(1, ids)
        merged = torch.where(sup_mask, torch.logaddexp(base, f_rec), base)
        return f_struct.scatter(1, ids, merged)

    # ── diagnostic: learned kernel shape ─────────────────────────────────────

    @torch.no_grad()
    def kernel(self, rel_id, ent_id, phases, device):
        self.eval()
        P = len(phases)
        ph = torch.as_tensor(phases, dtype=torch.float32, device=device)
        f = torch.zeros(1, P, N_FEAT, device=device)
        f[..., 0] = math.log1p(4.0)
        f[..., 2] = math.log1p(5.0)
        f[..., 7] = 1.0
        f[..., 8] = math.log1p(4.0)
        f[..., 1] = torch.log1p(ph * 5.0)
        f[..., 5] = ph; f[..., 11] = ph; f[..., 15] = ph
        rels = torch.tensor([rel_id], device=device)
        ids = torch.full((1, P), ent_id, device=device, dtype=torch.long)
        return self.recurrence(rels, ids, f)[0]
