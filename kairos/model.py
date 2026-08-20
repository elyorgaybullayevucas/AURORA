"""
KAIROS — phase-conditioned recurrence for temporal knowledge graph forecasting.

(kairos: the opportune moment. The model is about WHEN a fact is due.)

============================== WHERE THIS SITS ==============================
Two families currently hold the benchmarks, and they win for different
reasons:

  DaeMon (IJCAI'23) learns query-aware temporal PATH representations between
  the subject and each candidate. It dominates WIKI (82.38 MRR) and YAGO
  (91.59), where facts persist and recur.

  DiMNet (2025) evolves subgraph sequences with cross-time perception and
  DISENTANGLES node features into active and stable factors. It leads
  ICEWS18 (34.13) and GDELT (21.93). Its ablation is unambiguous about what
  carries the model: removing disentanglement costs 11.38 MRR, removing
  virtual-subgraph refinement 9.62, removing multi-span 4.97.

This model takes the backbone components that the literature has already
shown to matter -- multi-span evolution with cross-time carry, and
active/stable disentanglement -- and cites them as such. They are not the
contribution. They are the floor a new claim has to be tested on.

================================ THE CLAIM ==================================
Every recurrence mechanism in this literature scores a historical candidate
with

        s(o) = f(count_o) * g(dt_o),        g non-increasing in dt

  CyGNet, CENET, TiRGN, RE-GCN, DaeMon : g(dt) = exp(-lambda*dt), lambda fixed
  GAttNHP (2026)                       : g(dt) = exp(-gamma_u*dt), gamma learned

PROPOSITION. Let o* be the answer and o a distractor with dt_o <= dt_{o*} and
count_o >= count_{o*}. Then s(o) >= s(o*) for every non-decreasing f and
every non-increasing g. No decay rate, learned or fixed, and no reweighting
of counts ranks o* strictly first.

Such queries are MONOTONE-BLOCKED. Their rate is a property of the data, is
measured by diagnose.py before training, and upper-bounds what the whole
family can reach on the recurrent subset. They are produced by periodic
recurrence, refractory periods, and burst-then-die processes.

KAIROS sets

    log lambda_rec(o) = log sum_j w_j(r, o, phi_o) kappa_j(p_o),  w_j >= 0
    p_o = dt_o / mean_gap_o                                       (PHASE)

kappa a fixed RBF basis over phase. Non-monotone in dt by construction, so
the blocked queries become reachable. Fitting w_j to exp(-lambda*p) recovers
the classical term, so every scorer above is a special case up to basis
resolution; --phase_off restores exactly that special case as the ablation
that isolates the claim.

=============================== COMBINATION =================================
A query is a draw from a superposition of two marked point processes,

    lambda(o) = lambda_struct(o | G_<t, s, r) + lambda_rec(o | H_o, s, r)

Superposition adds intensities, so the branches combine through logaddexp.
Summing logits would multiply intensities, which has no point-process
meaning and is what let one branch silently rescale the other in earlier
iterations of this work. There is no mixing gate, so there is no gate to
collapse.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kairos.data import N_FEAT
from kairos.path_branch import PathBranch


# ── structural backbone ──────────────────────────────────────────────────────

class MultiSpanLayer(nn.Module):
    """
    One GNN layer over a snapshot, with a cross-time link to the SAME layer
    depth at the previous timestamp (DiMNet's multi-span idea). Without that
    link each snapshot is encoded independently and a node cannot perceive
    how its same-hop neighbourhood changed.

    Aggregation keeps mean and max, a reduced form of PNA; the two together
    separate "many weak neighbours" from "one decisive neighbour", which a
    mean alone cannot.
    """

    def __init__(self, d, dropout):
        super().__init__()
        self.w_msg = nn.Linear(d, d, bias=False)
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_cross = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, R, src, rel, dst, prev):
        msg = self.w_msg(H[src] + R[rel])
        # Accumulators must follow msg's dtype, not H's. Under autocast the
        # linear returns bf16 while the entity table stays fp32, and
        # index_add_ requires both operands to match exactly.
        dt, dev = msg.dtype, msg.device
        n, d = H.size(0), msg.size(1)

        mean = torch.zeros(n, d, device=dev, dtype=dt)
        mean.index_add_(0, dst, msg)
        deg = torch.zeros(n, 1, device=dev, dtype=dt)
        deg.index_add_(0, dst, torch.ones(dst.numel(), 1, device=dev, dtype=dt))
        mean = mean / deg.clamp(min=1.0)

        mx = torch.full((n, d), -1e4, device=dev, dtype=dt)
        mx = mx.index_reduce(0, dst, msg, "amax", include_self=True)
        mx = torch.where(deg > 0, mx, torch.zeros_like(mx))

        out = self.mix(torch.cat([mean, mx], -1)) + self.w_self(H)
        if prev is not None:
            out = out + self.w_cross(prev)
        return self.norm(self.drop(F.relu(out)))


class Disentangler(nn.Module):
    """
    Split a node state into an ACTIVE factor (what the neighbourhood is
    changing) and a STABLE factor (what the node is regardless of time).
    The two attention scores are complementary by construction, so the
    factors cannot both claim the same component.

    This is the single component DiMNet's ablation shows matters most
    (-11.38 MRR when removed), which is why it is in the backbone here.
    """

    def __init__(self, d):
        super().__init__()
        self.probe = nn.Linear(d, 2, bias=False)

    def forward(self, H):
        w = torch.softmax(self.probe(H), dim=-1)          # (N, 2)
        return H * w[:, :1], H * w[:, 1:]                 # active, stable


class Evolver(nn.Module):
    """Multi-span evolution with disentangled state carried across snapshots."""

    def __init__(self, d, omega, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiSpanLayer(d, dropout) for _ in range(omega)])
        self.dis = Disentangler(d)
        self.cell = nn.GRUCell(d, d)
        self.out = nn.Linear(2 * d, d)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, E0, R, history):
        """
        history: list of (src, rel, dst), oldest first.
        Returns (E, aux) where aux is the stable-factor drift penalty.
        """
        # `active` is created on first use so it picks up the autocast dtype
        # of the layer output rather than the fp32 embedding table.
        active = None
        stable = None
        prev_layers = [None] * len(self.layers)
        aux = E0.new_zeros(())
        n_steps = 0

        for (src, rel, dst) in history:
            if src.numel() == 0:
                continue
            H = E0 if stable is None else self.out(
                torch.cat([active, stable], -1))
            outs = []
            for l, layer in enumerate(self.layers):
                H = layer(H, R, src, rel, dst, prev_layers[l])
                outs.append(H)
            prev_layers = outs

            a, b = self.dis(H)
            if active is None:
                active = torch.zeros_like(a)
            active = self.cell(a, active)
            if stable is not None:
                # the stable factor should not drift between adjacent steps
                aux = aux + (1.0 - F.cosine_similarity(stable, b, dim=-1)).mean()
                n_steps += 1
            stable = b

        if stable is None:
            return E0, E0.new_zeros(())

        E = self.out(torch.cat([active, stable], -1))
        u = torch.sigmoid(self.gate(torch.cat([E, E0], -1)))
        E = u * E + (1 - u) * E0
        return E, aux / max(n_steps, 1)


class ConvTransE(nn.Module):
    """
    ConvTransE with LayerNorm in place of the canonical BatchNorm.

    BatchNorm is wrong for this training regime. Batches here are snapshots:
    the batch is however many facts occurred at one timestamp, which varies
    by more than an order of magnitude across timestamps (72 to 4650 on the
    datasets used here). Batch statistics are correspondingly unstable, and
    because long snapshots must be split into query chunks to fit in memory,
    BatchNorm also makes the result depend on the chunk size: the statistics
    are computed over whichever queries land in the same chunk. LayerNorm
    normalises per sample, so snapshot size and chunking
    change nothing. Whether BatchNorm would also cost accuracy here was not
    measured; it is replaced because its output depends on how the snapshot
    happens to be split, which makes the training objective ill-defined.
    """

    def __init__(self, d, channels, kernel, dropout):
        super().__init__()
        self.n0 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.n1 = nn.LayerNorm(d)
        self.fc = nn.Linear(channels * d, d)
        self.n2 = nn.LayerNorm(d)
        self.d0 = nn.Dropout(dropout)
        self.d1 = nn.Dropout(dropout)

    def forward(self, h_s, h_r, table, bias):
        B = h_s.size(0)
        x = self.d0(self.n0(torch.stack([h_s, h_r], 1)))
        x = self.d1(F.relu(self.n1(self.conv(x))))
        x = F.relu(self.n2(self.fc(x.reshape(B, -1))))
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
        # Isolates the phase SIGNAL from feature richness. --phase_off swaps
        # the whole branch for the published monotone form, which changes two
        # things at once: it removes the non-monotone basis AND cuts the
        # branch down from 16 features to (count, dt). This variant keeps the
        # trunk and every other feature and only blanks the three phase
        # entries, so the difference against `full` is attributable to phase.
        self.phase_feat_off = getattr(cfg, "phase_feat_off", False)
        self.path_off = getattr(cfg, "path_off", True)

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R2, d)
        self.ent_bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        self.evolver = Evolver(d, cfg.gcn_layers, cfg.dropout)
        self.decoder = ConvTransE(d, cfg.conv_channels, 3, cfg.dropout)

        # ── path branch: the third intensity, no entity embeddings ───────────
        self.path = None if self.path_off else PathBranch(
            cfg.path_dim, cfg.path_layers, R2, cfg.dropout)

        # ── recurrence kernel: the contribution ──────────────────────────────
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

        # ── the monotone baseline (--phase_off) ──────────────────────────────
        # This has to reproduce the published family EXACTLY:
        #     s(o) = f(count) * exp(-lambda * dt),  lambda >= 0
        # in log space,  log a + p*log1p(count) - b*dt  with p, b >= 0,
        # where a, p, b are predicted from (r, o) ONLY.
        #
        # An earlier version routed the full feature vector through the same
        # trunk and read out a scalar. That is not an ablation: the feature
        # vector contains the phase entries (5, 11, 15), so the MLP could
        # still learn a non-monotone function of phase. It measured "RBF
        # basis vs MLP on raw phase" -- both non-monotone -- and duly found
        # them equivalent (+0.16 MRR on YAGO), which says nothing about the
        # claim. Nothing temporal reaches this path now except count and dt.
        self.mono_trunk = nn.Sequential(
            nn.Linear(2 * dh, 2 * dh), nn.LayerNorm(2 * dh), nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.mono_head = nn.Linear(2 * dh, 3)
        # small but NOT zero: a zero weight matrix makes du/dz = 0 and the
        # trunk never receives gradient for the whole run
        nn.init.normal_(self.w_head.weight, std=0.02)
        nn.init.zeros_(self.w_head.bias)
        nn.init.normal_(self.mono_head.weight, std=0.02)
        nn.init.zeros_(self.mono_head.bias)
        self.rec_bias = nn.Parameter(torch.tensor(cfg.rec_bias_init))

    # ── branches ─────────────────────────────────────────────────────────────

    def evolve(self, history):
        if self.struct_off:
            return self.ent_emb.weight, self.ent_emb.weight.new_zeros(())
        return self.evolver(self.ent_emb.weight, self.rel_emb.weight, history)

    def structural(self, E, subs, rels):
        if self.struct_off:
            return self.ent_bias.unsqueeze(0).expand(subs.numel(), -1)
        return self.decoder(E[subs], self.rel_emb(rels), E, self.ent_bias)

    def recurrence(self, rels, sup_ids, sup_feat):
        B, S, _ = sup_feat.shape
        h_r = self.rel_ctx(self.rel_emb(rels)).unsqueeze(1).expand(B, S, -1)
        h_o = self.ent_ctx(self.ent_emb(sup_ids.clamp(0, self.N - 1)))

        if self.phase_off:
            # log a(r,o) + p(r,o)*log1p(count) - b(r,o)*dt,  p, b >= 0
            # monotone non-increasing in dt, non-decreasing in count:
            # the published family, with a learned per-(r,o) decay rate.
            z = self.mono_trunk(torch.cat([h_r, h_o], -1))
            log_a, p_raw, b_raw = self.mono_head(z).unbind(-1)
            cnt = sup_feat[..., 0]                      # log1p(count(s,r,o))
            dt = torch.expm1(sup_feat[..., 1]).clamp(min=0)
            return (log_a
                    + F.softplus(p_raw) * cnt
                    - F.softplus(b_raw) * dt) + self.rec_bias

        feat = sup_feat
        if self.phase_feat_off:
            feat = feat.clone()
            feat[..., 5] = 0.0        # phase (s,r,o)
            feat[..., 11] = 0.0       # phase (s,.,o)
            feat[..., 15] = 0.0       # phase (r,.,o)

        x = torch.cat([self.feat_norm(feat), h_r, h_o], -1)
        z = self.trunk(x)

        if self.phase_feat_off:
            # same trunk, same features minus phase, scalar readout
            return self.mono_head(z)[..., 0] + self.rec_bias

        w = F.softplus(self.w_head(z)).view(B, S, self.n_ctx, self.n_basis)
        p = torch.stack([sup_feat[..., 5], sup_feat[..., 11],
                         sup_feat[..., 15]], 2)
        c = self.centres.view(1, 1, 1, -1)
        wd = self.log_width.exp().view(1, 1, 1, -1).clamp(1e-2, 8.0)
        k = torch.exp(-0.5 * ((p.unsqueeze(-1) - c) / wd) ** 2)
        lam = (w * k).sum(-1).sum(-1)
        return torch.log(lam + 1e-8) + self.rec_bias

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, E, subs, rels, sup_ids, sup_feat, sup_mask,
                return_parts=False, history=None):
        f_struct = self.structural(E, subs, rels)

        # superpose the path intensity over every entity, before recurrence
        if self.path is not None and history is not None:
            f_path = self.path(subs, rels, history, self.N)
            f_struct = torch.logaddexp(f_struct, f_path)

        if self.rec_off:
            return (f_struct, f_struct) if return_parts else f_struct

        f_rec = self.recurrence(rels, sup_ids, sup_feat)
        f_rec = f_rec.masked_fill(~sup_mask, -1e4)

        ids = sup_ids.clamp(0, self.N - 1)
        base = f_struct.gather(1, ids)
        merged = torch.where(sup_mask, torch.logaddexp(base, f_rec), base)
        out = f_struct.scatter(1, ids, merged)
        # the structural scores are returned separately so they can be
        # supervised on their own; see the deep-supervision note in
        # train_kairos.py
        return (out, f_struct) if return_parts else out

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
