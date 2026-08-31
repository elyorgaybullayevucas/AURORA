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
import re

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


def config_of(tag):
    """
    The configuration a run belongs to, which is its tag minus the seed.

    Tags are written as <config><seed>, e.g. v2, v2s2, v2s3 are three seeds of
    configuration "v2"; ds0s2, ds0s3, ds0s7 are three seeds of "ds0"; s2, s3,
    s7 are three seeds of the untagged baseline configuration.
    """
    return re.sub(r"s\d+$", "", tag or "") or "-"


def load(save_dir="checkpoints"):
    """
    Group runs by (dataset, variant, configuration) and keep EVERY seed there.

    Two bugs have been fixed here, in order.

    Keying by variant alone silently kept only the last file loaded, so a
    five-seed sweep -- all of which write variant "full" -- was reported as a
    single run.

    Keying by (dataset, variant) then pooled runs that are not seeds of each
    other. On YAGO that put ten files under "full": three seeds of the current
    phase-free model, three of the earlier model that still had phase, three
    with deep supervision disabled, and the original. The reported 91.40+-0.48
    was therefore variation ACROSS CONFIGURATIONS presented as seed noise, and
    the conclusion drawn from it -- that we do not separate from DaeMon -- was
    not supported by it either. Seeds of one configuration are the only thing
    a standard deviation may be taken over.
    """
    runs = {}
    for f in sorted(glob.glob(os.path.join(save_dir, "*_kairos_*_results.json"))):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        tag = d.get("tag") or (d.get("config") or {}).get("tag") or ""
        cfgname = config_of(tag)
        key = d["variant"] if cfgname == "-" else f"{d['variant']} ({cfgname})"
        runs.setdefault(d["dataset"], {}).setdefault(key, []).append(d)
    return runs


# The configuration the paper reports. Older tags are kept on disk and still
# printed in the per-dataset tables, but they are superseded and must not be
# the ones a headline comparison or the claim test reads.
CURRENT = "v2"


def pick(byvariant, variant):
    """
    Entries for `variant` in the configuration we currently report.

    Prefers the current tag, then an untagged run, then any configuration of
    that variant -- and never merges two of them, which is the whole point.
    """
    for k in (f"{variant} ({CURRENT})", variant):
        if k in byvariant:
            return byvariant[k]
    for k in sorted(byvariant):
        if k == variant or k.startswith(variant + " ("):
            return byvariant[k]
    return []


def agg(entries, proto=None, key="MRR"):
    """mean, std, n over the seeds of one (dataset, variant)."""
    vals = []
    for d in entries:
        t = d["test"].get(proto) if proto else d["test"]
        if t and key in t:
            vals.append(t[key] * 100)
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5, len(vals)


def fmt(m, sd, n):
    if m is None:
        return "     -  "
    return f"{m:6.2f}" + (f"±{sd:.2f}" if n > 1 else "      ")


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
        for variant, entries in sorted(runs[ds].items()):
            cols = [fmt(*agg(entries, PROTO, k))
                    for k in ("MRR", "Hits@1", "Hits@3", "Hits@10")]
            n = agg(entries, PROTO, "MRR")[2]
            label = f"KAIROS {variant}" + (f" [{n} seeds]" if n > 1 else "")
            print(f"  {label:<26} " + " ".join(cols))

        fe = pick(runs[ds], "full")
        full = fe[0]["test"].get(PROTO) if fe else None
        ref = BASELINES.get(ds, {}).get(SOTA.get(ds))
        if full and ref:
            dm = agg(fe, PROTO, "MRR")[0] - ref[0]
            dh = agg(fe, PROTO, "Hits@1")[0] - ref[1]
            sd_m = agg(fe, PROTO, "MRR")[1]
            n_seeds = agg(fe, PROTO, "MRR")[2]
            if n_seeds > 1 and abs(dm) < sd_m:
                print(f"  => vs {SOTA[ds]}: MRR {dm:+.2f} but seed std is "
                      f"{sd_m:.2f} over {n_seeds} seeds -- NOT separated")
            verdict = "beats" if (dm > 0 and dh > 0) else \
                      "mixed" if (dm > 0 or dh > 0) else "below"
            print(f"  => vs {SOTA[ds]}: MRR {dm:+.2f}, H@1 {dh:+.2f}  [{verdict}]")

    # ── the claim: blocked vs clean across variants ──────────────────────────
    print(f"\n{'='*70}\n  CLAIM TEST — blocked vs clean, by variant\n{'='*70}")
    print("  The clean isolation is full vs no-phase-feature: same trunk,")
    print("  same 13 other features, only the phase entries blanked.")
    print("  full vs monotone-kernel also removes 14 features, so nothing it")
    print("  reports can be attributed to phase.\n")
    print(f"  {'dataset':<9} {'variant':<16} "
          f"{'blocked H@1':>12} {'clean H@1':>11} {'no_hist H@1':>12}")
    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        for variant, entries in sorted(runs.get(ds, {}).items()):
            t = entries[0]["test"]
            g = lambda k: (f"{t[k]['Hits@1']*100:.2f}" if k in t else "-")
            print(f"  {ds:<9} {variant:<16} {g('blocked'):>12} "
                  f"{g('clean'):>11} {g('no_history'):>12}")

    print(f"\n  {'dataset':<9} {'what is compared':<36} "
          f"{'blocked':>9} {'clean':>9}")
    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        fu = (pick(runs.get(ds, {}), "full") or [{}])[0].get("test", {})
        nf = (pick(runs.get(ds, {}), "no-phase-feature") or [{}])[0].get("test", {})
        mk = (pick(runs.get(ds, {}), "monotone-kernel") or [{}])[0].get("test", {})

        def delta(a, b, k):
            if k in a and k in b:
                return f"{(a[k]['Hits@1'] - b[k]['Hits@1']) * 100:+9.2f}"
            return f"{'-':>9}"

        if nf:
            print(f"  {ds:<9} {'phase alone (full - no_phase_feat)':<36} "
                  f"{delta(fu, nf, 'blocked')} {delta(fu, nf, 'clean')}")
        if mk:
            print(f"  {ds:<9} {'phase + 14 features (full - mono)':<36} "
                  f"{delta(fu, mk, 'blocked')} {delta(fu, mk, 'clean')}")

    print()
    print("  Where the first row is around zero or negative while the second")
    print("  is large, the gain belongs to the other temporal features and")
    print("  not to phase.")

    print(f"\n{'='*70}\n  OTHER PROTOCOLS (for comparison with papers that "
          f"report them)\n{'='*70}")
    print(f"  {'dataset':<9} {'variant':<16} {'raw MRR':>9} {'raw H@1':>9} "
          f"{'t-unaware MRR':>15}")
    for ds in ("YAGO", "WIKI", "ICEWS18", "GDELT"):
        for variant, entries in sorted(runs.get(ds, {}).items()):
            t = entries[0]["test"]
            r = t.get("raw", {}); u = t.get("time_unaware_filtered", {})
            print(f"  {ds:<9} {variant:<16} "
                  f"{r.get('MRR',0)*100:>9.2f} {r.get('Hits@1',0)*100:>9.2f} "
                  f"{u.get('MRR',0)*100:>15.2f}")
    print()


if __name__ == "__main__":
    main()
