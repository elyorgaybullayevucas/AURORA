#!/usr/bin/env python
"""
Prepare a LogCL checkout (github.com/WeiChen3690/LogCL) for a comparison
under OUR evaluation protocol.

    python external/patch_logcl.py /path/to/LogCL                 # as released
    python external/patch_logcl.py /path/to/LogCL_fix --fix-inverse

Run the two on SEPARATE copies of the checkout: the fix changes the
preprocessing output as well, and the two must not share it.

WHAT IT CHANGES
  1. src/hyperparameter_range.py is created as an empty stub. main.py imports
     hp_range from it; the released repository does not contain the file, so
     main.py does not start as released. hp_range is never used.
  2. main.py records the raw timestamp of every test snapshot (the released
     code keeps only the snapshot's position), because our filter is keyed on
     the real timestamp.
  3. In test mode, if LOGCL_DUMP is set, every test snapshot's full score
     matrix -- forward and inverse queries -- is written to
     $LOGCL_DUMP/part_<snapshot>.npz as float32, together with (s, r, t, o).
     Nothing else in LogCL's evaluation is touched; its own numbers still
     print, so the two can be put side by side.
     eval_external.py --dump $LOGCL_DUMP then ranks those scores with our
     filter and our tie handling, in our environment. The two codebases never
     share a Python process: LogCL needs torch 1.12 and DGL, ours does not.

WITH --fix-inverse
  get_sample_from_history_graph[3] builds the historical subgraph for the
  INVERSE queries around

        dst_set = set(triples[:, 0])

  An inverse query is (o, r^-1, ?) and its answer is s = triples[:, 0]. So the
  inverse subgraph is centred on the true answers: they enter message passing
  with their historical edges, other candidates do not. The same line appears
  in data/get_his_subg.py, so this holds at training time too. The query
  subjects of the inverse direction are triples[:, 2]; --fix-inverse uses
  those, in both files.

  Whether this changes LogCL's numbers, and by how much, is an empirical
  question. That is why the fix is a switch and both variants are run.

Idempotent: re-running on a patched checkout changes nothing.
"""
import argparse
import os
import sys

MARK = "# [aurora-bridge]"


def patch(path, old, new, what, required=True):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if new in s:
        print(f"  = {what} (already applied)")
        return
    if old not in s:
        if required:
            sys.exit(f"  ! {what}: anchor not found in {path}; the LogCL code "
                     f"differs from the version this patch was written for")
        print(f"  - {what}: not present, skipped")
        return
    s = s.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"  + {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logcl_dir")
    ap.add_argument("--fix-inverse", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.logcl_dir)
    mainpy = os.path.join(root, "src", "main.py")
    if not os.path.exists(mainpy):
        sys.exit(f"not a LogCL checkout: {mainpy} missing")
    print(f"patching {root}")

    stub = os.path.join(root, "src", "hyperparameter_range.py")
    if not os.path.exists(stub):
        with open(stub, "w") as f:
            f.write(f"{MARK} missing from the released repository; unused\n"
                    "hp_range = {}\n")
        print("  + src/hyperparameter_range.py stub")

    patch(mainpy,
          "    test_list = utils.split_by_time(data.test)\n",
          "    test_list = utils.split_by_time(data.test)\n"
          f"    {MARK} raw timestamps, in the order split_by_time emits them\n"
          "    global TEST_TIMES\n"
          "    TEST_TIMES = [int(x) for x in pd.unique(data.test[:, 3])]\n"
          "    assert len(TEST_TIMES) == len(test_list), (\n"
          "        'test.txt is not grouped by timestamp; snapshot order and '\n"
          "        'timestamps cannot be matched')\n",
          "record raw test timestamps")

    patch(mainpy,
          "        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)\n",
          "        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)\n"
          f"        {MARK} dump full score matrices for re-ranking under another protocol\n"
          "        if mode == 'test' and os.environ.get('LOGCL_DUMP'):\n"
          "            _d = os.environ['LOGCL_DUMP']\n"
          "            os.makedirs(_d, exist_ok=True)\n"
          "            _tr = torch.cat([test_triples, inv_test_triples]).cpu().numpy()\n"
          "            _sc = torch.cat([final_score, inv_final_score]).float().cpu().numpy()\n"
          "            _inv = np.r_[np.zeros(len(test_triples), bool), np.ones(len(inv_test_triples), bool)]\n"
          "            np.savez(os.path.join(_d, 'part_%05d.npz' % time_idx),\n"
          "                     sub=_tr[:, 0], rel=_tr[:, 1], obj=_tr[:, 2],\n"
          "                     t=np.full(len(_tr), TEST_TIMES[time_idx], np.int64),\n"
          "                     inverse=_inv, scores=_sc)\n",
          "dump test scores when LOGCL_DUMP is set")

    if a.fix_inverse:
        old = "    dst_set = set(triples[:, 0])\n"
        new = (f"    {MARK} inverse queries are (o, r^-1, ?): their subjects are\n"
               "    # triples[:, 2]. triples[:, 0] are the ANSWERS.\n"
               "    dst_set = set(triples[:, 2])\n")
        patch(mainpy, old, new, "fix inverse subgraph centre (src/main.py)")
        prep = os.path.join(root, "data", "get_his_subg.py")
        if os.path.exists(prep):
            patch(prep, old, new, "fix inverse subgraph centre (data/get_his_subg.py)")
        else:
            print("  ! data/get_his_subg.py not found: unzip data.zip first, "
                  "then re-run this patch, or training will use the "
                  "unfixed preprocessing")
    print("done")


if __name__ == "__main__":
    main()
