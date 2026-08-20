# What the reference implementations actually do

Read from source: DaeMon (IJCAI'23), CENET, GenTKG.

## DaeMon — `src/model.py`, `src/layers.py`

The important fact is what it does **not** have: **there are no entity
embeddings anywhere in the model.** The only embedding table is
`self.query = nn.Embedding(num_relation, input_dim)`.

Node states are *query-conditioned* and shaped `(batch, num_nodes, dim)`.
For a query `(s, r, ?)` the state is initialised to the relation embedding
at node `s` and zero everywhere else, then propagated over the snapshot:

```
initial_stat.scatter_add_(1, index, query)          # only s is seeded
message = input_j * relation_j                       # DistMult
aggregate = PNA(mean, max, min, std) x {1, log-deg, 1/log-deg}
```

So the representation of a candidate `o` is an aggregate of the *paths* from
`s` to `o` under relation `r`. Identity of `o` never enters. This is NBFNet
with a temporal memory: `output` of shape `(batch, nodes, dim)` is carried
across snapshots through a time-aware gate

```
initial_stat = g * initial_stat + (1 - g) * last_stat,  g = sigmoid(W last_stat)
```

Scoring is an MLP on `[path_state(o) ; query_r]`.

Other choices worth noting:
- `input_dim = 64`, `hidden_dims = [64, 64]` — the model is tiny.
- Loss is **not** a softmax over entities. It is binary cross-entropy over
  1 positive and 64 negatives, weighted by self-adversarial sampling
  (`softmax(neg_scores / 0.5)`), plus an orthogonality regulariser on the
  relation matrix.
- `batch_size = 32`, and the README recommends 4 GPUs — the
  `(batch, nodes, dim)` state is what forces the small batch.

## CENET — `cenet_model.py`

CENET's mechanism is an explicit **historical / non-historical decision**.
An `Oracle` MLP predicts, per query, whether the answer will be an entity
the subject has seen before. Two masks, `history_tag` and `non_history_tag`,
are built from the frequency vector with `+lambdax` / `-lambdax`, and they
push the scores of the two populations apart. Training combines a masked NCE
loss with a supervised contrastive loss on the query representation; at test
time the oracle's prediction selects which side to trust.

## GenTKG

LLM-based: retrieves temporal-logical rules, renders history as a prompt,
and fine-tunes a generative model. A different regime -- no shared
evaluation protocol with the above, so it is not a comparable baseline.

---

# What this says about our model

## 1. Our failure on `no_history` is architectural, not a tuning issue

Measured H@1 on the stratum where the answer has no `(s,r)` history:

| | YAGO | WIKI | ICEWS18 | GDELT |
|---|---|---|---|---|
| share of test | 7.3 % | 13.2 % | 50.1 % | 40.4 % |
| our H@1 | 0.79 | 1.22 | 5.65 | 1.63 |

Our structural branch scores candidates as `decoder(E[s], R[r]) · E[o]`. For
a novel fact this asks "is `o` the kind of entity that generally follows
`(s, r)`" — it can only answer from entity identity. DaeMon asks a different
question, "is there a relational path from `s` to `o` in the recent graph",
which is answerable without ever having seen `o` with `s`. That is why it
wins on exactly this population, and it is the largest single pool of
failures we have on the event datasets.

## 2. The recurrence branch is shortcutting training

`--rec_off` reaches 46.46 MRR on YAGO, so the structural branch *can* learn.
In the full model it does not: recurrence explains the loss almost
immediately on a 92.8 %-recurrent dataset, the gradient it leaves for the
structural branch is small, and training converges within a couple of epochs
with a structural branch that never became useful. The superposition is
correct as a likelihood, but a single objective on the sum lets the easy
branch absorb the signal.

The fix is deep supervision: score each branch on its own as well as jointly,

```
L = L(logaddexp(f_struct, f_rec)) + beta * L(f_struct)
```

so the structural branch is trained against the full label set regardless of
what recurrence is doing. This costs one extra cross-entropy and no new
parameters.

## 3. What not to copy

DaeMon's `(batch, nodes, dim)` state is why it runs at batch 32 on 4 GPUs.
Our training is snapshot-batched with thousands of queries per timestamp;
carrying a per-query state over all entities would be two orders of
magnitude more memory. A path branch here has to be restricted to a
candidate frontier rather than the full node set, or trained in the
per-query regime DaeMon uses. That is a real cost and should be decided
deliberately, not by default.
