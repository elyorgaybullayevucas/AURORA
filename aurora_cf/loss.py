"""
Loss functions for AURORA-CF.
  - LabelSmoothingCE: main 1-vs-N classification loss
  - InfoNCELoss: contrastive loss on neural logits only
  - AURORACFLoss: combined loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)
        nll  = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_prob.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class InfoNCELoss(nn.Module):
    """In-batch contrastive loss on neural query repr."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor,
                ent_emb: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query, dim=-1)
        e = F.normalize(ent_emb[targets], dim=-1)
        # In-batch: query vs positive entity of every other sample
        sim = q @ e.T / self.temperature        # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)


class AURORACFLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 alpha: float = 0.4,
                 temperature: float = 0.07):
        super().__init__()
        self.ce      = LabelSmoothingCE(smoothing)
        self.infonce = InfoNCELoss(temperature)
        self.alpha   = alpha

    def forward(self, logits: torch.Tensor, query: torch.Tensor,
                targets: torch.Tensor,
                ent_emb: torch.Tensor) -> dict:
        ce_loss  = self.ce(logits, targets)
        nce_loss = self.infonce(query, ent_emb, targets)
        total    = (1 - self.alpha) * ce_loss + self.alpha * nce_loss
        return {
            "_total":  total,
            "total":   total.item(),
            "ce":      ce_loss.item(),
            "infonce": nce_loss.item(),
        }
