"""Filtered evaluation for NHC, plus support-set diagnostics."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from aurora_cf.data_hazard import hazard_collate


@torch.no_grad()
def evaluate(model, loader, split: str = "valid",
             device: torch.device = torch.device("cpu"),
             batch_size: int = 512, hits_at: tuple = (1, 3, 10),
             verbose: bool = True) -> Dict[str, float]:

    model.eval()
    dataset = loader.valid_set if split == "valid" else loader.test_set
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=hazard_collate)

    mrr_sum = 0.0
    hits = {k: 0.0 for k in hits_at}
    in_sup = 0
    total = 0

    for batch in tqdm(dl, desc=f"Eval [{split}]",
                      disable=not verbose, dynamic_ncols=True):
        subs, rels, objs, times, sup_ids, sup_feat, sup_mask = batch
        subs = subs.to(device); rels = rels.to(device); objs = objs.to(device)
        sup_ids = sup_ids.to(device); sup_feat = sup_feat.to(device)
        sup_mask = sup_mask.to(device)

        logits, _ = model(subs, rels, sup_ids, sup_feat, sup_mask)

        for i in range(subs.size(0)):
            s, r, t = subs[i].item(), rels[i].item(), times[i].item()
            o_true = objs[i].item()
            sc = logits[i].clone()

            for o_f in loader.index.all_answers.get((s, r, t), set()):
                if o_f != o_true:
                    sc[o_f] = float("-inf")

            rank = (sc > sc[o_true]).sum().item() + 1
            mrr_sum += 1.0 / rank
            for k in hits_at:
                hits[k] += float(rank <= k)
            in_sup += float(
                (sup_ids[i][sup_mask[i]] == o_true).any().item())
            total += 1

    res = {"MRR": mrr_sum / total}
    res.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    res["SupportRecall"] = in_sup / total
    return res
