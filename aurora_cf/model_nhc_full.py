"""
NHC-Full — Neural Hazard Copy over a full-ranking temporal encoder.

Two branches, added in log-space (no gate, so nothing can collapse):

  STRUCTURAL BRANCH  (scores all N entities)
      relation-aware attention over the subject's local subgraph in each of
      the last H snapshots -> GRU across snapshots -> evolved subject state
      -> ConvTransE decoder against the full entity table.
      This is a standard, strong TKG backbone (RE-GCN / TiRGN family) and is
      responsible for recall: it can rank entities the subject has never
      touched.

  HAZARD BRANCH  (the contribution; scores the candidate set only)
      log lambda(o | H_o, s, r) = g_theta(phi(H_o), e_r, e_o)
      phi contains full inter-arrival statistics including PHASE = dt/mean_gap,
      expanded through an RBF basis. Prior copy mechanisms use
      f(count)*exp(-lambda*dt), which is monotone in dt and therefore cannot
      represent periodic or refractory recurrence. This branch is responsible
      for precision: it decides the ORDER of the historically plausible
      answers.

      score(o) = structural(o) + 1[o in S] * hazard(o)

Ablations the design supports directly:
      structural only          (hazard_off=True)
      structural + exp-decay   (phase_off=True, exp_only=True)
      structural + hazard      (full)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora_cf.data_full import N_FEAT


# ── structural branch ────────────────────────────────────────────────────────

class SnapshotEncoder(nn.Module):
    """Relation-aware attention over one snapshot's neighbours."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.d = d
        self.h = n_heads
        self.dk = d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d * 2, d)
        self.v = nn.Linear(d * 2, d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_s, h_ne, h_nr, mask):
        """
        h_s : (B, d)          subject state
        h_ne: (B, H, K, d)    neighbour entity embeddings
        h_nr: (B, H, K, d)    neighbour relation embeddings
        mask: (B, H, K)
        returns (B, H, d)
        """
        B, H, K, d = h_ne.shape
        msg = torch.cat([h_ne, h_nr], dim=-1)              # (B,H,K,2d)

        q = self.q(h_s).view(B, 1, self.h, self.dk)        # (B,1,h,dk)
        k = self.k(msg).view(B, H, K, self.h, self.dk)
        v = self.v(msg).view(B, H, K, self.h, self.dk)

        att = (q.unsqueeze(1) * k).sum(-1) / (self.dk ** 0.5)   # (B,H,K,h)
        att = att.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        # snapshots with no neighbours: keep them finite
        empty = (~mask).all(dim=2, keepdim=True).unsqueeze(-1)  # (B,H,1,1)
        att = torch.where(empty.expand_as(att), torch.zeros_like(att), att)
        att = torch.softmax(att, dim=2)
        att = torch.nan_to_num(att, nan=0.0)
        att = self.drop(att)

        z = (att.unsqueeze(-1) * v).sum(2)                 # (B,H,h,dk)
        z = z.reshape(B, H, d)
        z = self.out(z)
        z = z + h_s.unsqueeze(1)                           # residual
        return self.norm(z)


class ConvTransE(nn.Module):
    """Standard ConvTransE decoder — scores every entity."""

    def __init__(self, d, channels=64, kernel=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel,
                              padding=kernel // 2, bias=True)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(d)
        self.fc = nn.Linear(channels * d, d)
        self.drop = nn.Dropout(dropout)
        self.feat_drop = nn.Dropout(dropout)

    def forward(self, h_s, h_r, ent_table):
        B, d = h_s.shape
        x = torch.stack([h_s, h_r], dim=1)                 # (B,2,d)
        x = self.bn0(x)
        x = self.drop(x)
        x = F.relu(self.bn1(self.conv(x)))                 # (B,C,d)
        x = self.feat_drop(x)
        x = self.fc(x.view(B, -1))                         # (B,d)
        x = F.relu(self.bn2(x))
        return x @ ent_table.T                             # (B,N)


# ── full model ───────────────────────────────────────────────────────────────

class NHCFullModel(nn.Module):

    def __init__(self, num_entities, num_relations, cfg):
        super().__init__()
        d = cfg.embed_dim
        dh = cfg.hazard_dim
        R = num_relations * 2
        self.num_entities = num_entities
        self.hazard_off = getattr(cfg, "hazard_off", False)
        self.phase_off = getattr(cfg, "phase_off", False)

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        # ── structural ────────────────────────────────────────────────────────
        self.snap_enc = SnapshotEncoder(d, cfg.n_heads, cfg.dropout)
        self.gru = nn.GRU(d, d, num_layers=cfg.gru_layers,
                          batch_first=True,
                          dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.evolve_norm = nn.LayerNorm(d)
        self.decoder = ConvTransE(d, cfg.conv_channels, 3, cfg.dropout)

        # ── hazard ────────────────────────────────────────────────────────────
        centres = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25,
                                1.5, 2.0, 3.0, 4.0, 6.0])
        self.register_buffer("phase_centres", centres)
        self.phase_width = nn.Parameter(torch.full((len(centres),), 0.35))
        n_phase = 0 if self.phase_off else len(centres) * 3

        self.feat_norm = nn.LayerNorm(N_FEAT)
        self.rel_ctx = nn.Linear(d, dh)
        self.ent_ctx = nn.Linear(d, dh)
        self.hazard_net = nn.Sequential(
            nn.Linear(N_FEAT + n_phase + dh * 2, dh * 2),
            nn.LayerNorm(dh * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dh * 2, dh),
            nn.GELU(),
            nn.Linear(dh, 1),
        )
        # hazard starts at zero -> epoch 1 is exactly the structural baseline,
        # so any gain is attributable to the hazard branch.
        nn.init.zeros_(self.hazard_net[-1].weight)
        nn.init.zeros_(self.hazard_net[-1].bias)

        self.hazard_scale = nn.Parameter(torch.tensor(1.0))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _phase_basis(self, phase):
        c = self.phase_centres.view(1, 1, -1)
        w = F.softplus(self.phase_width).view(1, 1, -1) + 1e-3
        z = (phase.unsqueeze(-1) - c) / w
        return torch.exp(-0.5 * z * z)

    def encode(self, subs, ne, nr, nm):
        N = self.num_entities
        h_s = self.ent_emb(subs)
        h_ne = self.ent_emb(ne.clamp(0, N - 1))
        h_nr = self.rel_emb(nr.clamp(0, self.rel_emb.num_embeddings - 1))
        seq = self.snap_enc(h_s, h_ne, h_nr, nm)          # (B,H,d)
        _, hT = self.gru(seq)
        return self.evolve_norm(hT[-1] + h_s)             # (B,d)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, subs, rels, ne, nr, nm, sup_ids, sup_feat, sup_mask):
        N = self.num_entities
        h_r = self.rel_emb(rels)
        h_evo = self.encode(subs, ne, nr, nm)

        logits = self.decoder(h_evo, h_r, self.ent_emb.weight)   # (B,N)

        if self.hazard_off:
            return logits, h_evo

        f = self.feat_norm(sup_feat)
        parts = [f]
        if not self.phase_off:
            parts.append(self._phase_basis(sup_feat[..., 7]))
            parts.append(self._phase_basis(sup_feat[..., 8]))
            parts.append(self._phase_basis(sup_feat[..., 14]))

        B, S, _ = sup_feat.shape
        parts.append(self.rel_ctx(h_r).unsqueeze(1).expand(B, S, -1))
        parts.append(self.ent_ctx(self.ent_emb(sup_ids.clamp(0, N - 1))))

        hz = self.hazard_net(torch.cat(parts, -1)).squeeze(-1)
        hz = hz * self.hazard_scale
        hz = hz.masked_fill(~sup_mask, 0.0)

        logits = logits.scatter_add(1, sup_ids.clamp(0, N - 1),
                                    hz * sup_mask.float())
        return logits, h_evo

    # ── diagnostic: learned hazard vs phase ──────────────────────────────────

    @torch.no_grad()
    def hazard_curve(self, rel_id, ent_id, phases, device):
        self.eval()
        P = len(phases)
        ph = torch.as_tensor(phases, dtype=torch.float32,
                             device=device).view(1, P)
        feat = torch.zeros(1, P, N_FEAT, device=device)
        feat[..., 0] = float(torch.log1p(torch.tensor(4.0)))
        feat[..., 1] = float(torch.log1p(torch.tensor(4.0)))
        feat[..., 4] = float(torch.log1p(torch.tensor(5.0)))
        feat[..., 9] = 1.0
        feat[..., 7] = ph
        feat[..., 8] = ph
        feat[..., 14] = ph
        feat[..., 2] = torch.log1p(ph * 5.0)
        feat[..., 3] = torch.log1p(ph * 5.0)

        parts = [self.feat_norm(feat)]
        if not self.phase_off:
            parts += [self._phase_basis(feat[..., 7]),
                      self._phase_basis(feat[..., 8]),
                      self._phase_basis(feat[..., 14])]
        rels = torch.tensor([rel_id], device=device)
        ids = torch.full((1, P), ent_id, device=device, dtype=torch.long)
        parts.append(self.rel_ctx(self.rel_emb(rels)).unsqueeze(1).expand(1, P, -1))
        parts.append(self.ent_ctx(self.ent_emb(ids)))
        return (self.hazard_net(torch.cat(parts, -1)).squeeze(-1)[0]
                * self.hazard_scale)
