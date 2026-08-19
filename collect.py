#!/usr/bin/env python
"""
Collect every finished run and compare against the published numbers.

    python collect.py

Reads checkpoints/*_results.json, prints one table per dataset under the
time-aware filtered protocol (the one the baselines use), then the ablation
table and the stratified table that tests the claim.
"""
import glob
import json
import os

# time-aware filtered, x100. Sources recorded in TARGETS.md.
BASELINES = {
    "ICEWS18": {"RE-GCN": (30.58, 21.01), "DaeMon": (31.85, 22.67),
                "TiPNN": (32.17, 22.74), "DiMNet": (34.13, 23.29)},
    "GDELT":   {"RE-GCN": (19.64, 12.42), "DaeMon": (20.73, 13.65),
                "TiPNN": (21.17, 14.03), "DiMNet": (21.93, 14.03)},
    "WIKI":    {"RE-GCN": (77.55, 73.75), "TITer": (75.50, 72.96),
                "DaeMon": (82.38, 78.26)},
    "YAGO":    {"RE-GCN": (84.12, 80.76), "TITer": (87.47, 84.89),
                "DaeMon": (91.59, 90.03)},
}
SOTA = {"ICEWS18": "DiMNet", "GDELT": "DiMNet", "WIKI": "DaeMon",
        "YAGO": "DaeMon"}
PROTO = "time_aware_filtered"


def load(save_dir="checkpoints"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(save_dir, "*_kairos_*_results.json"))):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        runs.setdefault(d["dataset"], {})[d["variant"]] = d
    return runs


def pct(x):
    return f"{x*100:6.2f}"


def main():
    runs = load()
    if not runs:
        print("no finished runs in checkpoints/")
        return

    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        if ds not in runs:
            continue
        print(f"\n{'='*70}\n  {ds}   (time-aware filtered, x100)\n{'='*70}")
        print(f"  {'method':<22} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7}")
        for name, (mrr, h1) in BASELINES.get(ds, {}).items():
            star = "  <- SOTA" if name == SOTA.get(ds) else ""
            print(f"  {name:<22} {mrr:>7.2f} {h1:>7.2f} {'':>7} {'':>7}{star}")
        print("  " + "-" * 66)
        for variant, d in sorted(runs[ds].items()):
            t = d["test"].get(PROTO)
            if not t:
                continue
            print(f"  {'KAIROS ' + variant:<22} {pct(t['MRR'])} "
                  f"{pct(t['Hits@1'])} {pct(t['Hits@3'])} {pct(t['Hits@10'])}")

        full = runs[ds].get("full", {}).get("test", {}).get(PROTO)
        ref = BASELINES.get(ds, {}).get(SOTA.get(ds))
        if full and ref:
            dm = full["MRR"] * 100 - ref[0]
            dh = full["Hits@1"] * 100 - ref[1]
            verdict = "beats" if (dm > 0 and dh > 0) else \
                      "mixed" if (dm > 0 or dh > 0) else "below"
            print(f"  => vs {SOTA[ds]}: MRR {dm:+.2f}, H@1 {dh:+.2f}  [{verdict}]")

    # ── the claim: blocked vs clean across variants ──────────────────────────
    print(f"\n{'='*70}\n  CLAIM TEST — blocked vs clean, by variant\n{'='*70}")
    print("  If the phase basis is doing the work, dropping it (--phase_off)")
    print("  must cost far more on 'blocked' than on 'clean'. Similar losses")
    print("  on both would mean the basis was only extra capacity.\n")
    print(f"  {'dataset':<9} {'variant':<16} "
          f"{'blocked H@1':>12} {'clean H@1':>11} {'no_hist H@1':>12}")
    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        for variant, d in sorted(runs.get(ds, {}).items()):
            t = d["test"]
            g = lambda k: (f"{t[k]['Hits@1']*100:.2f}" if k in t else "-")
            print(f"  {ds:<9} {variant:<16} {g('blocked'):>12} "
                  f"{g('clean'):>11} {g('no_history'):>12}")

    for ds in runs:
        f = runs[ds].get("full", {}).get("test", {})
        p = runs[ds].get("monotone-kernel", {}).get("test", {})
        if "blocked" in f and "blocked" in p and "clean" in f and "clean" in p:
            db = (f["blocked"]["Hits@1"] - p["blocked"]["Hits@1"]) * 100
            dc = (f["clean"]["Hits@1"] - p["clean"]["Hits@1"]) * 100
            print(f"\n  {ds}: phase basis is worth {db:+.2f} H@1 on blocked "
                  f"and {dc:+.2f} on clean")
            if db > dc + 0.5:
                print("       -> consistent with the claim")
            else:
                print("       -> NOT consistent with the claim; the basis is "
                      "not acting where the argument says it should")

    print(f"\n{'='*70}\n  OTHER PROTOCOLS (for comparison with papers that "
          f"report them)\n{'='*70}")
    print(f"  {'dataset':<9} {'variant':<16} {'raw MRR':>9} {'raw H@1':>9} "
          f"{'t-unaware MRR':>15}")
    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        for variant, d in sorted(runs.get(ds, {}).items()):
            t = d["test"]
            r = t.get("raw", {}); u = t.get("time_unaware_filtered", {})
            print(f"  {ds:<9} {variant:<16} "
                  f"{r.get('MRR',0)*100:>9.2f} {r.get('Hits@1',0)*100:>9.2f} "
                  f"{u.get('MRR',0)*100:>15.2f}")
    print()


if __name__ == "__main__":
    main()
