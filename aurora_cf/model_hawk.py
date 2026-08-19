"""
HAWK — Hazard-Aware Knowledge forecasting.

================================ FORMULATION ================================
A query (s, r, ?, t) is treated as a draw from a SUPERPOSITION of two marked
point processes on the entity space:

    lambda(o | t) = lambda_nov(o | G_{<t}, s, r)  +  lambda_rec(o | H_o, s, r)

  lambda_nov  structural intensity, defined for every entity. Produced by a
              time-aware relational encoder over the recent snapshots and a
              ConvTransE decoder. Carries recall.

  lambda_rec  recurrence intensity, defined only on the candidate set S of
              entities with observable history. Carries precision.

For a superposition of point processes the probability that the next mark is
o is exactly lambda(o)/sum_o' lambda(o'), so the softmax cross-entropy on

    score(o) = log lambda(o) = logaddexp( f_nov(o), f_rec(o) )      (o in S)
             = f_nov(o)                                             (o not in S)

is the correct discrete-choice likelihood of the superposed process.

Two consequences, both of which we needed empirically:
  * Superposition ADDS INTENSITIES, i.e. combines in log-space through
    logaddexp, not through a sum of logits (a sum of logits multiplies
    intensities, which has no point-process meaning and let one branch
    silently rescale the other).
  * There is no mixing gate, so there is no gate to collapse. Earlier
    variants with a learned scalar or per-query gate all collapsed to a
    single branch within a few epochs.

================================ THE NOVELTY ================================
Every published TKG copy/recurrence mechanism (CyGNet, CENET, TiRGN, RE-GCN,
DaeMon) scores a historical candidate with

        s(o) = f(count_o) * exp(-lambda * dt_o)

PROPOSITION. Such a score is monotone non-increasing in dt_o for every fixed
count. Therefore no choice of lambda or f can produce a ranking in which a
candidate with larger dt outranks an otherwise identical candidate with
smaller dt.

That is exactly the behaviour required by periodic recurrence (a fact due
after a typical gap), by refractory recurrence (a fact that just fired and
will not fire again immediately), and by burst-then-die processes.

HAWK parameterises the recurrence intensity as

    f_rec(o) = log sum_j w_j(r, o, phi) * kappa_j(phase_o),   w_j >= 0
    phase_o  = dt_o / mean_gap_o

with kappa_j a fixed RBF basis over phase and w_j predicted per candidate.
Because the basis is non-monotone in phase, the induced hazard is
non-monotone in dt -- the class of rankings excluded by the proposition
becomes reachable. Setting the basis aside (--phase_off) recovers a
monotone-only model and is reported as an ablation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora_cf.tkg_index import TKGIndex

N_FEAT = TKGIndex.N_FEAT


# ── time encoding ────────────────────────────────────────────────────────────

class Time2Vec(nn.Module):
    """Learned Fourier features of elapsed time."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim - 1) * 0.1)
        self.b = nn.Parameter(torch.zeros(dim - 1))
        self.w0 = nn.Parameter(torch.tensor(0.01))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, dt):
        lin = self.w0 * dt.unsqueeze(-1) + self.b0
        per = torch.sin(dt.unsqueeze(-1) * self.w + self.b)
        return torch.cat([lin, per], dim=-1)


# ── structural encoder ───────────────────────────────────────────────────────

class TemporalRelAttention(nn.Module):
    """Multi-head attention over (neighbour entity, relation, elapsed time)."""

    def __init__(self, d, dt_dim, n_heads, dropout):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d * 2 + dt_dim, d * 2)
        self.out = nn.Linear(d, d)
        self.norm_in = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, d))

    def forward(self, h_q, h_ne, h_nr, h_dt, mask):
        B, H, K, d = h_ne.shape
        x = self.norm_in(h_q)
        q = self.q(x).view(B, 1, 1, self.h, self.dk)
        kv = self.kv(torch.cat([h_ne, h_nr, h_dt], -1))
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, H, K, self.h, self.dk)
        v = v.view(B, H, K, self.h, self.dk)

        att = (q * k).sum(-1) / math.sqrt(self.dk)            # (B,H,K,h)
        att = att.masked_fill(~mask.unsqueeze(-1), -1e4)
        empty = (~mask).all(2, keepdim=True).unsqueeze(-1)
        att = torch.softmax(att, dim=2)
        att = torch.where(empty.expand_as(att), torch.zeros_like(att), att)
        z = (self.drop(att).unsqueeze(-1) * v).sum(2).reshape(B, H, d)
        z = h_q.unsqueeze(1) + self.out(z)
        return self.norm_out(z + self.ffn(z))


class ConvTransE(nn.Module):
    def __init__(self, d, channels, kernel, dropout):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(2)
        self.conv = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc = nn.Linear(channels * d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, h_s, h_r, table, bias):
        B = h_s.size(0)
        x = self.d1(self.bn0(torch.stack([h_s, h_r], 1)))
        x = self.d2(F.relu(self.bn1(self.conv(x))))
        x = F.relu(self.bn2(self.fc(x.view(B, -1))))
        return x @ table.T + bias


# ── HAWK ─────────────────────────────────────────────────────────────────────

class HAWK(nn.Module):

    def __init__(self, num_entities, num_relations, cfg):
        super().__init__()
        d, dh = cfg.embed_dim, cfg.hazard_dim
        R = num_relations * 2
        self.N = num_entities
        self.hazard_off = cfg.hazard_off
        self.phase_off = cfg.phase_off
        self.struct_off = getattr(cfg, "struct_off", False)

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        self.ent_bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        # ── structural branch ────────────────────────────────────────────────
        self.t2v = Time2Vec(cfg.dt_dim)
        self.layers = nn.ModuleList([
            TemporalRelAttention(d, cfg.dt_dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)])
        self.gru = nn.GRU(d, d, num_layers=cfg.gru_layers, batch_first=True,
                          dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.evo_norm = nn.LayerNorm(d)
        self.decoder = ConvTransE(d, cfg.conv_channels, 3, cfg.dropout)

        # ── recurrence branch ────────────────────────────────────────────────
        centres = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                                2.0, 2.5, 3.0, 4.0, 6.0, 9.0])
        self.register_buffer("centres", centres)
        self.log_width = nn.Parameter(torch.full((len(centres),),
                                                 math.log(0.35)))
        self.n_basis = len(centres)
        self.n_ctx = 3                     # phases: (s,r), (s,.), (r,.)

        self.feat_norm = nn.LayerNorm(N_FEAT)
        self.rel_ctx = nn.Linear(d, dh)
        self.ent_ctx = nn.Linear(d, dh)
        trunk_in = N_FEAT + dh * 2
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, dh * 2), nn.LayerNorm(dh * 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dh * 2, dh * 2), nn.LayerNorm(dh * 2), nn.GELU(),
        )
        # non-negative mixture weights over the phase basis
        self.w_head = nn.Linear(dh * 2, self.n_basis * self.n_ctx)
        # monotone-only fallback path (used when --phase_off)
        self.mono_head = nn.Linear(dh * 2, 1)
        # Do NOT zero the weights here. With W = 0 the gradient reaching the
        # trunk is (dL/du)(du/dz) with du/dz = W = 0, so the whole recurrence
        # trunk is frozen at initialisation and never learns. Inertness at
        # step 0 is supplied by rec_bias instead, which does not block any
        # gradient path.
        nn.init.normal_(self.w_head.weight, std=0.02)
        nn.init.zeros_(self.w_head.bias)
        nn.init.normal_(self.mono_head.weight, std=0.02)
        nn.init.zeros_(self.mono_head.bias)

        # log lambda_rec = log(sum_j w_j kappa_j) + rec_bias
        #
        # rec_bias sets where the recurrence intensity starts relative to the
        # structural one. It is learnable and initialised low so that early
        # training is dominated by the structural branch, but NOT so low that
        # the branch is starved: the gradient reaching f_rec through
        # logaddexp is sigmoid(f_rec - f_nov), which vanishes if the gap is
        # made large. -4 keeps that factor around 2e-2 while the initial
        # perturbation of the structural scores stays under 0.05 nats.
        self.rec_bias = nn.Parameter(torch.tensor(cfg.rec_bias_init))

    # ── phase basis ──────────────────────────────────────────────────────────

    def _basis(self, phase):
        """phase (B,S) -> (B,S,C) non-negative RBF activations."""
        c = self.centres.view(1, 1, -1)
        w = self.log_width.exp().view(1, 1, -1).clamp(1e-2, 5.0)
        z = (phase.unsqueeze(-1) - c) / w
        return torch.exp(-0.5 * z * z)

    # ── branches ─────────────────────────────────────────────────────────────

    def _structural(self, subs, rels, ne, nr, ndt, nm):
        h_s = self.ent_emb(subs)
        h_ne = self.ent_emb(ne.clamp(0, self.N - 1))
        h_nr = self.rel_emb(nr.clamp(0, self.rel_emb.num_embeddings - 1))
        h_dt = self.t2v(ndt)
        h = h_s
        for layer in self.layers:
            seq = layer(h, h_ne, h_nr, h_dt, nm)      # (B,H,d)
            h = seq[:, 0]
        _, hT = self.gru(seq)
        h_evo = self.evo_norm(hT[-1] + h_s)
        return self.decoder(h_evo, self.rel_emb(rels),
                            self.ent_emb.weight, self.ent_bias)

    def _recurrence(self, rels, sup_ids, sup_feat, sup_mask):
        B, S, _ = sup_feat.shape
        x = torch.cat([
            self.feat_norm(sup_feat),
            self.rel_ctx(self.rel_emb(rels)).unsqueeze(1).expand(B, S, -1),
            self.ent_ctx(self.ent_emb(sup_ids.clamp(0, self.N - 1))),
        ], -1)
        z = self.trunk(x)

        if self.phase_off:
            # monotone-only: no phase basis, decay must come from raw features
            f_rec = self.mono_head(z).squeeze(-1)
        else:
            w = F.softplus(self.w_head(z)).view(B, S, self.n_ctx, self.n_basis)
            ph = torch.stack([sup_feat[..., 5],       # (s,r) phase
                              sup_feat[..., 11],      # (s,.) phase
                              sup_feat[..., 15]], 2)  # (r,.) phase
            k = torch.stack([self._basis(ph[..., i])
                             for i in range(self.n_ctx)], 2)   # (B,S,ctx,C)
            lam = (w * k).sum(-1).sum(-1)                       # (B,S) >= 0
            f_rec = torch.log(lam + 1e-8)

        return f_rec + self.rec_bias

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, subs, rels, ne, nr, ndt, nm, sup_ids, sup_feat, sup_mask):
        if self.struct_off:
            f_nov = self.ent_bias.unsqueeze(0).expand(subs.size(0), -1).clone()
        else:
            f_nov = self._structural(subs, rels, ne, nr, ndt, nm)

        if self.hazard_off:
            return f_nov

        f_rec = self._recurrence(rels, sup_ids, sup_feat, sup_mask)
        f_rec = f_rec.masked_fill(~sup_mask, -1e4)

        # superposition: lambda = lambda_nov + lambda_rec  ->  logaddexp
        ids = sup_ids.clamp(0, self.N - 1)
        base = f_nov.gather(1, ids)                       # (B,S)
        merged = torch.logaddexp(base, f_rec)
        merged = torch.where(sup_mask, merged, base)
        return f_nov.scatter(1, ids, merged)

    # ── diagnostic: learned hazard vs phase ──────────────────────────────────

    @torch.no_grad()
    def hazard_curve(self, rel_id, ent_id, phases, device):
        self.eval()
        P = len(phases)
        ph = torch.as_tensor(phases, dtype=torch.float32, device=device)
        feat = torch.zeros(1, P, N_FEAT, device=device)
        feat[..., 0] = math.log1p(4.0)
        feat[..., 2] = math.log1p(5.0)
        feat[..., 7] = 1.0
        feat[..., 8] = math.log1p(4.0)
        feat[..., 1] = torch.log1p(ph * 5.0)
        feat[..., 5] = ph
        feat[..., 11] = ph
        feat[..., 15] = ph
        rels = torch.tensor([rel_id], device=device)
        ids = torch.full((1, P), ent_id, device=device, dtype=torch.long)
        mask = torch.ones(1, P, dtype=torch.bool, device=device)
        return self._recurrence(rels, ids, feat, mask)[0]
