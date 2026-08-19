"""
Fork-safe, allocation-free temporal index for TKG forecasting.

Everything is stored as flat numpy arrays in CSR form. There are no Python
dicts on the query path, so DataLoader workers can fork without triggering
copy-on-write blow-up, and features can be computed ON THE FLY instead of
being materialised for every training sample.

That matters at GDELT scale: materialising per-candidate features for
3.46M augmented training samples would need ~10 GB of host RAM, which is
what made the precompute pipeline unusable on the largest dataset.

Query cost is O(log n + |S|) with all per-candidate statistics computed by
vectorised numpy reductions -- no Python loop over candidates.
"""
import numpy as np


class CSRGroup:
    """
    Groups (o, t) pairs by an integer key.

    Within a key block, entries are sorted by (o, t). This layout lets every
    per-candidate temporal statistic be produced by a single reduceat call.
    """

    __slots__ = ("keys", "start", "end", "obj", "time")

    def __init__(self, key, obj, time):
        order = np.lexsort((time, obj, key))
        key = key[order]; self.obj = obj[order].astype(np.int32)
        self.time = time[order].astype(np.int64)
        self.keys, starts = np.unique(key, return_index=True)
        self.start = starts.astype(np.int64)
        self.end = np.append(starts[1:], len(key)).astype(np.int64)

    def block(self, k):
        """Return (obj_slice, time_slice) for key k, or (None, None)."""
        i = np.searchsorted(self.keys, k)
        if i >= len(self.keys) or self.keys[i] != k:
            return None, None
        return self.obj[self.start[i]:self.end[i]], \
               self.time[self.start[i]:self.end[i]]

    def stats(self, k, t_q):
        """
        Temporal statistics of every candidate under key k, using only
        events strictly before t_q.

        Returns (ids, count, dt, mean_gap, gap_std, span) as numpy arrays,
        restricted to candidates with at least one event before t_q.
        Fully vectorised: no loop over candidates.
        """
        o, t = self.block(k)
        if o is None:
            z = np.zeros(0)
            return (np.zeros(0, np.int32), z, z, z, z, z)

        # group boundaries within the block (o is sorted)
        bnd = np.flatnonzero(np.diff(o)) + 1
        g_start = np.concatenate(([0], bnd))
        g_end = np.concatenate((bnd, [len(o)]))
        ids = o[g_start]

        # events before t_q, per group (t is sorted inside each group)
        before = (t < t_q).astype(np.int64)
        cnt = np.add.reduceat(before, g_start)
        keep = cnt > 0
        if not keep.any():
            z = np.zeros(0)
            return (np.zeros(0, np.int32), z, z, z, z, z)

        ids = ids[keep]
        g_start_k = g_start[keep]
        cnt_k = cnt[keep].astype(np.float64)

        # last / first observed time before t_q
        last = t[g_start_k + cnt_k.astype(np.int64) - 1].astype(np.float64)
        first = t[g_start_k].astype(np.float64)

        dt = t_q - last
        span = last - first
        # mean of consecutive gaps telescopes to (last-first)/(m-1)
        m1 = np.maximum(cnt_k - 1.0, 1.0)
        mean_gap = np.where(cnt_k >= 2, span / m1, 0.0)

        # dispersion of gaps: reduceat over squared diffs, masking group edges
        if len(t) > 1:
            d = np.diff(t).astype(np.float64)
            valid = np.ones(len(d), dtype=bool)
            valid[g_end[:-1] - 1] = False           # cross-group boundaries
            # only gaps fully before t_q
            valid &= (t[1:] < t_q)
            d2 = np.where(valid, d * d, 0.0)
            dv = np.where(valid, d, 0.0)
            s1 = np.add.reduceat(np.append(dv, 0.0), g_start)[keep]
            s2 = np.add.reduceat(np.append(d2, 0.0), g_start)[keep]
            n_g = np.maximum(cnt_k - 1.0, 1.0)
            var = np.maximum(s2 / n_g - (s1 / n_g) ** 2, 0.0)
            gap_std = np.where(cnt_k >= 3, np.sqrt(var), 0.0)
        else:
            gap_std = np.zeros_like(dt)

        return (ids.astype(np.int32), cnt_k, dt, mean_gap, gap_std, span)


class TKGIndex:
    """Temporal index over a quadruple set, with inverse edges included."""

    def __init__(self, quads, num_entities, num_relations,
                 step=1, use_inverse=True, rel_topk=64):
        self.step = max(int(step), 1)
        self.num_entities = int(num_entities)
        self.R2 = int(num_relations) * 2
        self.rel_topk = int(rel_topk)

        q = quads.astype(np.int64)
        if use_inverse:
            inv = np.stack([q[:, 2], q[:, 1] + num_relations,
                            q[:, 0], q[:, 3]], axis=1)
            q = np.concatenate([q, inv], axis=0)
        s, r, o, t = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        self.n_edges = len(q)

        # (s, r) -> (o, t)
        self.g_sr = CSRGroup(s * self.R2 + r, o, t)
        # (s, .) -> (o, t)
        self.g_s = CSRGroup(s.copy(), o, t)
        # (r, .) -> (o, t)
        self.g_r = CSRGroup(r.copy(), o, t)
        # (t, s) -> (o, r)   structural neighbourhoods; "obj"=o, we keep r too
        self.g_ts = CSRGroup(t * self.num_entities + s, o, t)
        # relation ids aligned with g_ts ordering
        order = np.lexsort((t, o, t * self.num_entities + s))
        self.ts_rel = r[order].astype(np.int32)

        # filtered-setting answers: (s, r, t) -> objects
        self.ans_key = s * self.R2 * 0 + (s * self.R2 + r)
        ak = (s * self.R2 + r).astype(np.int64) * (t.max() + 2) + t
        ord2 = np.argsort(ak, kind="stable")
        self.ans_k = ak[ord2]
        self.ans_o = o[ord2].astype(np.int32)
        self._t_span = int(t.max() + 2)

        # top-K globally frequent objects per relation
        self.rel_top = {}
        for rid in np.unique(r):
            ob, _ = self.g_r.block(int(rid))
            if ob is None:
                continue
            u, c = np.unique(ob, return_counts=True)
            k = min(self.rel_topk, len(u))
            self.rel_top[int(rid)] = u[np.argpartition(-c, k - 1)[:k]] \
                .astype(np.int32)
        self._empty = np.zeros(0, dtype=np.int32)

    # ── filtered answers ─────────────────────────────────────────────────────

    def answers(self, s, r, t):
        k = (int(s) * self.R2 + int(r)) * self._t_span + int(t)
        lo = np.searchsorted(self.ans_k, k, "left")
        hi = np.searchsorted(self.ans_k, k, "right")
        return self.ans_o[lo:hi]

    # ── structural neighbourhood ─────────────────────────────────────────────

    def neighbors(self, sub, t_q, H, K, rng):
        ne = np.zeros((H, K), dtype=np.int32)
        nr = np.zeros((H, K), dtype=np.int32)
        ndt = np.zeros((H, K), dtype=np.float32)
        nm = np.zeros((H, K), dtype=bool)

        t = t_q - self.step
        for h in range(H):
            if t < 0:
                break
            key = t * self.num_entities + int(sub)
            i = np.searchsorted(self.g_ts.keys, key)
            if i < len(self.g_ts.keys) and self.g_ts.keys[i] == key:
                a, b = self.g_ts.start[i], self.g_ts.end[i]
                n = b - a
                if n > K:
                    sel = rng.choice(n, K, replace=False) + a
                else:
                    sel = np.arange(a, b)
                m = len(sel)
                ne[h, :m] = self.g_ts.obj[sel]
                nr[h, :m] = self.ts_rel[sel]
                ndt[h, :m] = (t_q - t) / self.step
                nm[h, :m] = True
            t -= self.step
        return ne, nr, ndt, nm

    # ── candidate set + temporal features ────────────────────────────────────

    N_FEAT = 16

    def candidates(self, sub, rel, t_q, max_support):
        sub, rel, t_q = int(sub), int(rel), int(t_q)
        step = self.step

        i_sr, c_sr, d_sr, m_sr, sd_sr, sp_sr = \
            self.g_sr.stats(sub * self.R2 + rel, t_q)
        i_s, c_s, d_s, m_s, _, _ = self.g_s.stats(sub, t_q)
        i_r, c_r, d_r, m_r, _, _ = self.g_r.stats(rel, t_q)

        top = self.rel_top.get(rel, self._empty)
        ids = np.unique(np.concatenate([i_sr, i_s, top]).astype(np.int32)) \
            if (len(i_sr) or len(i_s) or len(top)) else self._empty
        ids = ids[ids < self.num_entities]
        if len(ids) == 0:
            return self._empty, np.zeros((0, self.N_FEAT), np.float32)

        def align(src_ids, *cols):
            """Map per-source statistics onto the unified candidate order."""
            out = [np.zeros(len(ids)) for _ in cols]
            if len(src_ids) == 0:
                return out
            pos = np.searchsorted(src_ids, ids)
            pos_c = np.clip(pos, 0, len(src_ids) - 1)
            hit = src_ids[pos_c] == ids
            for j, col in enumerate(cols):
                out[j][hit] = col[pos_c[hit]]
            return out

        c1, d1, m1, s1, p1 = align(i_sr, c_sr, d_sr, m_sr, sd_sr, sp_sr)
        c2, d2, m2 = align(i_s, c_s, d_s, m_s)
        c3, d3, m3 = align(i_r, c_r, d_r, m_r)

        d1 /= step; m1 /= step; s1 /= step; p1 /= step
        d2 /= step; m2 /= step
        d3 /= step; m3 /= step

        def phase(dt, mg, cnt):
            return np.where((mg > 1e-6) & (cnt > 0),
                            np.minimum(dt / np.maximum(mg, 1e-6), 20.0), 0.0)

        F = np.empty((len(ids), self.N_FEAT), dtype=np.float32)
        F[:, 0] = np.log1p(c1)
        F[:, 1] = np.log1p(np.maximum(d1, 0))
        F[:, 2] = np.log1p(m1)
        F[:, 3] = np.log1p(s1)
        F[:, 4] = np.log1p(np.maximum(p1, 0))
        F[:, 5] = phase(d1, m1, c1)
        F[:, 6] = c1 / (p1 + 1.0)
        F[:, 7] = (c1 > 0).astype(np.float32)
        F[:, 8] = np.log1p(c2)
        F[:, 9] = np.log1p(np.maximum(d2, 0))
        F[:, 10] = np.log1p(m2)
        F[:, 11] = phase(d2, m2, c2)
        F[:, 12] = np.log1p(c3)
        F[:, 13] = np.log1p(np.maximum(d3, 0))
        F[:, 14] = np.log1p(m3)
        F[:, 15] = phase(d3, m3, c3)

        if len(ids) > max_support:
            # subject-specific evidence first, then recency
            prio = -(F[:, 0] + F[:, 8]) * 10.0 + F[:, 1]
            keep = np.argpartition(prio, max_support - 1)[:max_support]
            ids, F = ids[keep], F[keep]

        return ids, F
