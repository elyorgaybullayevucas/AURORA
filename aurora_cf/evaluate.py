"""Filtered evaluation: MRR, Hits@1, Hits@3, Hits@10."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from aurora_cf.model import AURORACFModel
from aurora_cf.data import CFDataLoader, cf_collate


@torch.no_grad()
def evaluate(model: AURORACFModel, loader: CFDataLoader,
             split: str = "valid", device: torch.device = torch.device("cpu"),
             batch_size: int = 256, hits_at: tuple = (1, 3, 10),
             verbose: bool = True) -> Dict[str, float]:

    model.eval()
    dataset = loader.valid_set if split == "valid" else loader.test_set
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=cf_collate)

    mrr_sum = 0.0
    hits    = {k: 0.0 for k in hits_at}
    total   = 0
    N       = loader.num_entities

    for batch in tqdm(dl, desc=f"Eval [{split}]",
                      disable=not verbose, dynamic_ncols=True):
        subs, rels, objs, times, ne, nr, nm, rel_copy, ent_copy = batch

        subs = subs.to(device); rels = rels.to(device)
        objs = objs.to(device)
        ne   = ne.to(device);   nm   = nm.to(device)

        if rel_copy.shape[1] < N:
            rel_copy = torch.cat([rel_copy,
                torch.zeros(rel_copy.shape[0], N - rel_copy.shape[1])], dim=1)
        if ent_copy.shape[1] < N:
            ent_copy = torch.cat([ent_copy,
                torch.zeros(ent_copy.shape[0], N - ent_copy.shape[1])], dim=1)
        rel_copy = rel_copy.to(device)
        ent_copy = ent_copy.to(device)

        logits, _ = model(subs, rels, ne, nr, nm, rel_copy, ent_copy)

        for i in range(subs.size(0)):
            s, r, t  = subs[i].item(), rels[i].item(), times[i].item()
            o_true   = objs[i].item()
            sc       = logits[i].clone()

            for o_f in loader.index.all_answers.get((s, r, t), set()):
                if o_f != o_true:
                    sc[o_f] = float("-inf")

            rank = (sc > sc[o_true]).sum().item() + 1
            mrr_sum += 1.0 / rank
            for k in hits_at:
                hits[k] += float(rank <= k)
            total += 1

    results = {"MRR": mrr_sum / total}
    results.update({f"Hits@{k}": hits[k] / total for k in hits_at})
    return results
