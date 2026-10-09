"""
CADENCE — recurrence intensity from full inter-arrival statistics.

============================== WHERE THIS SITS ==============================
Two families hold the benchmarks, for different reasons. DaeMon learns
query-aware temporal PATH representations and leads WIKI (82.38 MRR) and
YAGO (91.59). DiMNet evolves subgraph sequences with cross-time perception
and disentangles node features into active and stable factors, leading
ICEWS18 (34.13) and GDELT (21.93); its ablation attributes -11.38 MRR to
disentanglement, -9.62 to virtual-subgraph refinement and -4.97 to
multi-span.

This model takes multi-span evolution and active/stable disentanglement as
backbone and cites them as prior work. They are not the contribution; they
are the floor a new claim has to be tested on.

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

Such queries are MONOTONE-BLOCKED. Their rate is a property of the data,
measured by diagnose.py before any training: 41.6% of YAGO, 29.3% of WIKI,
26.6% of ICEWS18 and 15.8% of GDELT test queries.

The published form is, in the end, a scorer over TWO features. This model
replaces it with a learned function of thirteen inter-arrival statistics --
counts, recency, mean gap, gap dispersion, span and rate, over the (s,r),
(s,.) and (r,.) histories -- which is not confined to be monotone in dt.
--phase_off restores the two-feature monotone form as the ablation.

=========================== A HYPOTHESIS THAT FAILED ========================
An earlier version placed a radial basis over PHASE = dt / mean_gap at the
centre of the model, on the strength of a real measurement: the median phase
of the true answer on YAGO and WIKI is exactly 1.000, so facts recur at close
to their own mean waiting time, and a monotone kernel peaks at phase 0.

The mechanism does not survive its own ablation. Blanking only the phase
inputs, holding the trunk and the other thirteen features fixed, moves H@1 on
the blocked stratum by -0.09 (YAGO), -0.52 (WIKI), -0.13 (ICEWS18) and -0.11
(GDELT) -- zero or slightly better without it, on all four. WIKI is the
sharpest case: median phase exactly 1.000, the largest gap to the monotone
form of any dataset, and still 0.52 better with phase removed. What that gap
measures is the other thirteen features.

Phase is therefore gone from the model, and the measurement is reported as a
negative result rather than as a mechanism.

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
import numpy as np
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


# ── global history context ──────────────────────────────────────────────────

class GlobalHistory(nn.Module):
    """
    Every entity's FULL observed history before t, summarised in one hop.

    Stratifying LogCL's ICEWS18 ranks under our own protocol put the whole
    MRR gap between the two models in one place. Where the answer has a
    history with the subject, this model leads by 17 MRR; where it has none
    -- half of all queries -- LogCL leads by 7.8, and that alone outweighs
    everything else. Those answers are, almost by definition, entities that
    have not interacted with the subject recently, and the structural branch
    sees only the last H snapshots: an entity quiet over that window enters
    the decoder with nothing but its static embedding. LogCL's advantage is
    a global history encoder.

    This is the smallest version of that idea that fits the existing model:
    for each entity o, the mean over every fact (o, r, x, t') with t' < t of
    E0[x] + R[r] -- all of its observed neighbours and how it met them, over
    the whole timeline, not a window -- projected and added to the initial
    state the evolver starts from and gates back to. It is one index_add over
    the facts before t, chunked so that GDELT's millions of edges never sit
    in memory at once, and it is order-free, so shuffled training is fine.
    Only facts strictly before the query timestamp are read.

    The projection starts at zero, so a fresh model with this enabled is
    exactly the model without it, and the data decides how much to use.
    """

    CHUNK = 200_000

    def __init__(self, d, layers=1):
        super().__init__()
        # One hop summarises an entity's own history. A second hop brings in
        # its neighbours' histories, which is the only route by which an entity
        # that never met the subject can be related to it -- and two-hop
        # reachability covers 77% of the out-of-support answers on ICEWS18.
        self.layers = layers
        self.mix = nn.ModuleList([nn.Linear(d, d) for _ in range(layers - 1)])
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _hop(self, X, R, src, rel, dst, k, deg):
        # The accumulator must have the dtype X[dst] + R[rel] promotes to.
        # Under bf16 autocast the second hop's X comes out of a Linear in
        # bf16 while the relation table stays fp32, so the message is fp32
        # and an accumulator in X's dtype makes index_add refuse it. The same
        # mismatch was fixed in MultiSpanLayer at the start of this project.
        acc = X.new_zeros(X.size(0), X.size(1),
                          dtype=torch.promote_types(X.dtype, R.dtype))
        for a in range(0, k, self.CHUNK):
            b = min(a + self.CHUNK, k)
            acc = acc.index_add(0, src[a:b], X[dst[a:b]] + R[rel[a:b]])
        return acc / deg.clamp(min=1.0).unsqueeze(1).to(acc.dtype)

    def forward(self, E0, R, src, rel, dst, k):
        n = E0.size(0)
        deg = torch.bincount(src[:k], minlength=n).to(E0.dtype)
        h = self._hop(E0, R, src, rel, dst, k, deg)
        for lin in self.mix:
            # residual: the second hop refines the first, never replaces it
            h = h + self._hop(torch.tanh(lin(h)), R, src, rel, dst, k, deg)
        return self.proj(self.ln(h)) * (deg > 0).unsqueeze(1).to(h.dtype)


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
        self.path_off = getattr(cfg, "path_off", True)

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R2, d)
        self.ent_bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        self.evolver = Evolver(d, cfg.gcn_layers, cfg.dropout)
        self.decoder = ConvTransE(d, cfg.conv_channels, 3, cfg.dropout)
        self.glob = (GlobalHistory(d, getattr(cfg, "global_layers", 1))
                     if getattr(cfg, "global_hist", False) else None)
        self._tl = None              # (src, rel, dst, t) sorted by t; set_timeline

        # ── path branch: the third intensity, no entity embeddings ───────────
        self.path = None if self.path_off else PathBranch(
            cfg.path_dim, cfg.path_layers, R2, cfg.dropout)

        # ── recurrence intensity ─────────────────────────────────────────────
        # A learned function of the full inter-arrival statistics: counts,
        # recency, mean gap, gap dispersion, span and rate, over the (s,r),
        # (s,.) and (r,.) histories.
        #
        # The three PHASE entries (indices 5, 11, 15 = dt / mean_gap) are
        # deliberately excluded. An earlier version of this work put a radial
        # basis over phase here and claimed it as the mechanism. Blanking the
        # phase entries alone, holding the trunk and all other features fixed,
        # changes H@1 on the blocked stratum by -0.09 (YAGO), -0.52 (WIKI),
        # -0.13 (ICEWS18) and -0.11 (GDELT) -- zero or slightly in favour of
        # dropping it, on all four. The gain attributed to phase belongs to
        # the other thirteen features, so the basis and the phase inputs are
        # gone and the model is smaller for it.
        keep = [i for i in range(N_FEAT) if i not in (5, 11, 15)]
        self.register_buffer("feat_idx", torch.tensor(keep, dtype=torch.long))
        n_feat = len(keep)

        # ── query conditioning ───────────────────────────────────────────────
        # lambda_rec is written lambda_rec(o | H_o, s, r), but an earlier
        # version of this code passed only (r, o, H_o): the subject never
        # reached the head. That omission has a specific and measurable cost.
        # The branches combine by logaddexp, and logaddexp(a, b) >= a, so the
        # recurrence intensity can only ever raise a historical candidate
        # relative to a non-historical one -- it has no way to say "for this
        # query the history is uninformative". Without the subject the head
        # cannot even represent the distinction: the features are counts of
        # (s,r,o), but nothing tells it which subject regime it is in.
        #
        # That is exactly where the model is weakest. On the stratum where the
        # answer never occurred with (s,r) -- 50.1% of ICEWS18 and 40.4% of
        # GDELT -- H@1 is 5.65 and 1.63. Conditioning on the evolved subject
        # state lets the intensity be driven down for such queries. It is not
        # a gate between the branches: no mixing weight is learned, and the
        # superposition rule is untouched. It is the intensity depending on
        # the conditioning information the model was always defined to use.
        self.sub_ctx = nn.Linear(d, dh)
        self.query_off = getattr(cfg, "query_off", False)
        q_extra = 0 if self.query_off else dh

        self.feat_norm = nn.LayerNorm(n_feat)
        self.rel_ctx = nn.Linear(d, dh)
        self.ent_ctx = nn.Linear(d, dh)
        self.trunk = nn.Sequential(
            nn.Linear(n_feat + 2 * dh + q_extra, 2 * dh), nn.LayerNorm(2 * dh),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * dh, 2 * dh), nn.LayerNorm(2 * dh), nn.GELU(),
        )
        self.rec_head = nn.Linear(2 * dh, 1)

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
        nn.init.normal_(self.rec_head.weight, std=0.02)
        nn.init.zeros_(self.rec_head.bias)
        nn.init.normal_(self.mono_head.weight, std=0.02)
        nn.init.zeros_(self.mono_head.bias)
        self.rec_bias = nn.Parameter(torch.tensor(cfg.rec_bias_init))

    # ── branches ─────────────────────────────────────────────────────────────

    def set_timeline(self, edges_by_t, device):
        """
        Hand the model every fact of the timeline, sorted by time, for the
        global history context. Facts at or after a query's timestamp are
        never read: evolve() cuts the arrays at searchsorted(t, 'left').
        """
        ts = sorted(edges_by_t)
        cat = lambda i: torch.from_numpy(
            np.concatenate([edges_by_t[t][i] for t in ts]).astype(np.int64))
        tt = np.concatenate([np.full(len(edges_by_t[t][0]), t, np.int64)
                             for t in ts])
        self._tl = tuple(x.to(device) for x in
                         (cat(0), cat(1), cat(2), torch.from_numpy(tt)))

    def evolve(self, history, t=None):
        if self.struct_off:
            return self.ent_emb.weight, self.ent_emb.weight.new_zeros(())
        E0 = self.ent_emb.weight
        if self.glob is not None and self._tl is not None and t is not None:
            src, rel, dst, tt = self._tl
            k = int(torch.searchsorted(tt, torch.tensor(int(t), device=tt.device),
                                       right=False))
            if k > 0:
                E0 = E0 + self.glob(E0, self.rel_emb.weight, src, rel, dst, k)
        return self.evolver(E0, self.rel_emb.weight, history)

    def structural(self, E, subs, rels):
        if self.struct_off:
            return self.ent_bias.unsqueeze(0).expand(subs.numel(), -1)
        return self.decoder(E[subs], self.rel_emb(rels), E, self.ent_bias)

    def recurrence(self, rels, sup_ids, sup_feat, h_sub=None):
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

        feat = sup_feat.index_select(-1, self.feat_idx)
        parts = [self.feat_norm(feat), h_r, h_o]
        if not self.query_off:
            if h_sub is None:
                # kernel() and any caller without an evolved table: fall back
                # to the static embedding so the shape is still correct.
                h_sub = self.sub_ctx(self.ent_emb.weight.new_zeros(B, self.sub_ctx.in_features))
            parts.append(h_sub.unsqueeze(1).expand(B, S, -1))
        z = self.trunk(torch.cat(parts, -1))
        return self.rec_head(z).squeeze(-1) + self.rec_bias

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, E, subs, rels, sup_ids, sup_feat, sup_mask,
                return_parts=False, history=None):
        f_struct = self.structural(E, subs, rels)

        # superpose the path intensity over every entity, before recurrence
        f_path = None
        if self.path is not None and history is not None:
            f_path = self.path(subs, rels, history, self.N)
            f_struct = torch.logaddexp(f_struct, f_path)

        if self.rec_off:
            return ((f_struct, f_struct, f_path) if return_parts
                    else f_struct)

        h_sub = None if self.query_off else self.sub_ctx(E[subs])
        f_rec = self.recurrence(rels, sup_ids, sup_feat, h_sub)
        f_rec = f_rec.masked_fill(~sup_mask, -1e4)

        ids = sup_ids.clamp(0, self.N - 1)
        base = f_struct.gather(1, ids)
        merged = torch.where(sup_mask, torch.logaddexp(base, f_rec), base)
        out = f_struct.scatter(1, ids, merged)
        # the branch scores are returned separately so each can be supervised
        # on its own; see the deep-supervision note in train_kairos.py
        return (out, f_struct, f_path) if return_parts else out

    # ── diagnostic: learned kernel shape ─────────────────────────────────────

    @torch.no_grad()
    def kernel(self, rel_id, ent_id, dts, device, count=4.0, mean_gap=5.0):
        """
        Learned recurrence intensity as a function of elapsed time, with the
        count and the mean inter-arrival gap held fixed.

        This used to sweep phase, which stopped being meaningful when the
        phase inputs were removed. The question it answers is unchanged and is
        the one Proposition 1 is about: is the learned intensity monotone in
        elapsed time? The published form is monotone by construction; a
        non-monotone curve here is what places this model outside that family.
        """
        self.eval()
        P = len(dts)
        dt = torch.as_tensor(dts, dtype=torch.float32, device=device)
        f = torch.zeros(1, P, N_FEAT, device=device)
        f[..., 0] = math.log1p(count)          # (s,r,o) count
        f[..., 1] = torch.log1p(dt)            # (s,r,o) recency
        f[..., 2] = math.log1p(mean_gap)       # mean inter-arrival
        f[..., 4] = math.log1p(mean_gap * count)   # observed span
        f[..., 7] = 1.0                        # has (s,r) history
        f[..., 8] = math.log1p(count)          # (s,.,o) count
        f[..., 9] = torch.log1p(dt)            # (s,.,o) recency
        rels = torch.tensor([rel_id], device=device)
        ids = torch.full((1, P), ent_id, device=device, dtype=torch.long)
        return self.recurrence(rels, ids, f)[0]
