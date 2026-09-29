"""
PRISM — Partitioned Routing of Intensities over Support Marks.

================================ WHY A NEW MODEL ============================
Thirty runs of CADENCE and its ablations left three measured facts.

  1. An unconstrained recurrence intensity works. Restricting it to the
     monotone form costs +5.3 / +18.6 H@1 on the blocked stratum of YAGO /
     WIKI. That part is kept unchanged.
  2. Every variant collapses on the same stratum: queries whose answer is not
     in the candidate support S (50% of ICEWS18, H@1 ~6). A path branch did
     not lift it, and conditioning the recurrence on the subject did not
     either (no_history H@1 moved by at most +0.2).
  3. The structural branch is not weak in isolation (46.5 MRR on YAGO alone).
     It is starved in joint training: recurrence explains the loss on the
     easy queries first, and forcing structure with an auxiliary loss hurt.

Fact 3 explains fact 2. Under a superposition both branches share ONE
softmax over all N entities, so every gradient step on a recurrent query also
pushes the structural scores -- and recurrent queries are the majority. The
structural branch is trained mostly on a problem it is not needed for.

============================== THE LIKELIHOOD ===============================
Partition the entities by the query's support S (the candidate set built from
the observed history). The answer lies in S or in its complement, and which
one is known at training time. Write

    P(o | q) = pi(q)       * p_S(o | q)        o in S
             = (1 - pi(q)) * p_N(o | q)        o not in S

where p_S is a softmax over S only, p_N a softmax over the complement only,
and pi a router. This is an exact probability distribution over all N
entities, and its log-likelihood FACTORISES:

    log P(o|q) = [o in S] (log pi + log p_S(o))
               + [o notin S] (log(1-pi) + log p_N(o))

  = a binary routing term  +  a cross-entropy on exactly one of two disjoint
    supports.

Consequences, each of which answers one of the facts above:

  * p_N receives gradient ONLY from queries whose answer is outside S. The
    structural expert is trained on the no-history problem and nothing else,
    so it can no longer be starved by the recurrent majority (fact 3).
  * p_S is where CADENCE's scoring lives, unchanged: structure and the
    unconstrained recurrence intensity superposed by logaddexp, now normalised
    over S. Proposition 1 applies to it exactly as before (fact 1).
  * The router is supervised directly by the observable label [o in S]. The
    gated models that collapsed earlier had no such label; this one cannot
    drift onto one expert without paying the routing cross-entropy for it.
  * Ranking compares pi * p_S(o) against (1-pi) * p_N(o'), two calibrated
    probabilities, instead of two logits on unrelated scales.

CADENCE's logaddexp could only ever RAISE a support candidate relative to a
non-support one. Here mass is moved in both directions, by a quantity trained
to be the probability that the answer is historical at all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kairos.model import KAIROS

NEG = -1e4          # finite "minus infinity": stays finite under bf16 autocast


class PRISM(KAIROS):
    """
    Reuses CADENCE's backbone (multi-span evolver, ConvTransE decoder) and its
    recurrence intensity verbatim. What is new is the likelihood and the
    router; the components are inherited so an ablation against CADENCE
    changes exactly that and nothing else.
    """

    def __init__(self, num_entities, num_relations, cfg):
        super().__init__(num_entities, num_relations, cfg)
        d, dh = cfg.embed_dim, cfg.hazard_dim
        n_feat = self.feat_idx.numel()
        # router input: subject state, relation, and a pooled summary of the
        # support (mean and max of the normalised statistics, and its size)
        self.r_sub = nn.Linear(d, dh)
        self.r_rel = nn.Linear(d, dh)
        self.router = nn.Sequential(
            nn.Linear(2 * dh + 2 * n_feat + 1, 2 * dh), nn.LayerNorm(2 * dh),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * dh, 1))

    # ── router ───────────────────────────────────────────────────────────────

    def route(self, E, subs, rels, sup_feat, sup_mask):
        """Logit of pi(q) = P(answer in S | q)."""
        feat = self.feat_norm(sup_feat.index_select(-1, self.feat_idx))
        m = sup_mask.unsqueeze(-1).to(feat.dtype)
        n = m.sum(1).clamp(min=1.0)
        mean = (feat * m).sum(1) / n
        mx = feat.masked_fill(~sup_mask.unsqueeze(-1), NEG).amax(1)
        mx = torch.where(sup_mask.any(1, keepdim=True), mx,
                         torch.zeros_like(mx))
        size = torch.log1p(sup_mask.sum(1, keepdim=True).to(feat.dtype))
        x = torch.cat([self.r_sub(E[subs]), self.r_rel(self.rel_emb(rels)),
                       mean, mx, size], -1)
        rho = self.router(x).squeeze(-1)
        # an empty support means the answer cannot be historical: pi = 0
        return rho.masked_fill(~sup_mask.any(1), -30.0)

    # ── forward: log P(o | q) over all entities, exactly normalised ─────────

    def forward(self, E, subs, rels, sup_ids, sup_feat, sup_mask,
                return_parts=False, history=None):
        B = subs.numel()
        f_struct = self.structural(E, subs, rels).float()          # (B, N)
        ids = sup_ids.clamp(0, self.N - 1)

        # Support membership over all N. Written with index_put on the VALID
        # (row, slot) pairs only: padding slots are clamped to entity 0, and a
        # scatter with duplicate indices has no defined write order, so a
        # padding write could silently erase a real entity 0 in the support.
        rows, slots = sup_mask.nonzero(as_tuple=True)
        ents = ids[rows, slots]
        in_S = torch.zeros(B, self.N, dtype=torch.bool, device=f_struct.device)
        in_S[rows, ents] = True

        # p_S: CADENCE's scoring, restricted to S and normalised there
        h_sub = None if self.query_off else self.sub_ctx(E[subs])
        f_rec = self.recurrence(rels, sup_ids, sup_feat, h_sub).float()
        base = f_struct.gather(1, ids)
        s_S = torch.logaddexp(base, f_rec).masked_fill(~sup_mask, NEG)
        logp_S = torch.log_softmax(s_S, dim=1)                     # (B, S)

        # p_N: structure only, normalised over the complement of S
        logp_N = torch.log_softmax(f_struct.masked_fill(in_S, NEG), dim=1)

        rho = self.route(E, subs, rels, sup_feat, sup_mask).float()
        log_pi, log_1mpi = F.logsigmoid(rho), F.logsigmoid(-rho)

        out = log_1mpi.unsqueeze(1) + logp_N
        on_S = log_pi.unsqueeze(1) + logp_S
        out = out.index_put((rows, ents), on_S[rows, slots])
        if return_parts:
            return out, None, None
        return out
