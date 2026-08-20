"""
Path branch — query-conditioned relational propagation over snapshots.

WHY THIS EXISTS
Measured H@1 on the stratum where the answer has never appeared for the
query's (s, r) pair:

    YAGO 0.79 | WIKI 1.22 | ICEWS18 5.65 | GDELT 1.63
    share of test: 7.3 % | 13.2 % | 50.1 % | 40.4 %

Half of ICEWS18 and two fifths of GDELT sit in that stratum, essentially
unanswered. The recurrence branch is silent there by construction, and the
structural branch scores `decoder(E[s], R[r]) . E[o]`, which can only ask
"is o the kind of entity that generally follows (s, r)" -- a question about
entity identity. For a fact that has never occurred, identity is the wrong
question.

DaeMon (IJCAI'23) asks a different one, and it is why it leads YAGO and
WIKI: is there a relational PATH from s to o in the recent graph? Its model
contains no entity embeddings at all. Node states are query-conditioned,
seeded with the relation embedding at s and zero elsewhere, then propagated;
a candidate's representation is an aggregate of the paths reaching it, so a
never-seen candidate is still scored on evidence rather than on identity.

This module reimplements that mechanism as a third intensity, so the three
combine as a superposition:

    lambda(o) = lambda_struct(o) + lambda_rec(o) + lambda_path(o)

COST, STATED PLAINLY
The state is (queries, num_nodes, dim). That is what forces DaeMon to
batch 32 across 4 GPUs, and it is ~50x the arithmetic of our other two
branches. Snapshots are gradient-checkpointed so memory holds one state per
snapshot instead of one per layer per snapshot, and queries are processed in
small chunks. Expect roughly an order of magnitude more time per epoch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PathLayer(nn.Module):
    """
    One propagation step. Messages are DistMult products with a relation
    vector projected from the query, so the same edge carries different
    information depending on what was asked. Aggregation keeps mean and max
    with a degree scaling, a reduced PNA.
    """

    def __init__(self, d, num_relations, dropout):
        super().__init__()
        self.d = d
        self.R = num_relations
        self.rel_proj = nn.Linear(d, num_relations * d)
        self.update = nn.Linear(d * 3, d)      # [input, mean, max]
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, q, seed, src, rel, dst, N):
        """
        h    : (B, N, d)  current state
        q    : (B, d)     query relation embedding
        seed : (B, N, d)  boundary condition, re-added every layer
        """
        B, _, d = h.shape
        rel_mat = self.rel_proj(q).view(B, self.R, d)          # (B, R, d)
        r_e = rel_mat[:, rel]                                   # (B, E, d)
        msg = h[:, src] * r_e                                   # DistMult

        mean = torch.zeros(B, N, d, device=h.device, dtype=msg.dtype)
        mean.index_add_(1, dst, msg)
        deg = torch.zeros(N, 1, device=h.device, dtype=msg.dtype)
        deg.index_add_(0, dst, torch.ones(dst.numel(), 1, device=h.device,
                                          dtype=msg.dtype))
        mean = mean / deg.clamp(min=1.0).unsqueeze(0)

        mx = torch.full((B, N, d), -1e4, device=h.device, dtype=msg.dtype)
        mx = mx.index_reduce(1, dst, msg, "amax", include_self=True)
        mx = torch.where(deg.unsqueeze(0) > 0, mx, torch.zeros_like(mx))

        out = self.update(torch.cat([h, mean, mx], -1))
        return self.norm(self.drop(F.relu(out)) + seed)


class PathBranch(nn.Module):
    """
    Propagates a query-conditioned state across the history snapshots and
    scores every entity from it. Contains no entity embeddings.
    """

    def __init__(self, d, n_layers, num_relations, dropout):
        super().__init__()
        self.d = d
        self.query_emb = nn.Embedding(num_relations, d)
        nn.init.normal_(self.query_emb.weight, std=0.05)
        self.layers = nn.ModuleList(
            [PathLayer(d, num_relations, dropout) for _ in range(n_layers)])
        self.mem_gate = nn.Linear(d, d)
        self.score = nn.Sequential(
            nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, 1))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def _step(self, state, q, subs, src, rel, dst, N, first):
        B = q.size(0)
        seed = torch.zeros(B, N, self.d, device=q.device, dtype=q.dtype)
        seed[torch.arange(B, device=q.device), subs] = q
        if not first:
            g = torch.sigmoid(self.mem_gate(state))
            seed = g * seed + (1 - g) * state
        h = seed
        for layer in self.layers:
            h = layer(h, q, seed, src, rel, dst, N)
        return h

    def forward(self, subs, rels, history, N, use_checkpoint=True):
        """
        subs, rels : (B,)   keep B small; the state is (B, N, d)
        history    : list of (src, rel, dst), oldest first
        returns    : (B, N) log-intensity over every entity
        """
        q = self.query_emb(rels)
        state = torch.zeros(q.size(0), N, self.d, device=q.device,
                            dtype=q.dtype)
        first = True
        for (src, rel, dst) in history:
            if src.numel() == 0:
                continue
            if use_checkpoint and self.training:
                state = checkpoint(self._step, state, q, subs, src, rel, dst,
                                   N, first, use_reentrant=False)
            else:
                state = self._step(state, q, subs, src, rel, dst, N, first)
            first = False

        qx = q.unsqueeze(1).expand(-1, N, -1)
        return self.score(torch.cat([state, qx], -1)).squeeze(-1) + self.bias
