# PC-NN-v4

PC-NN is used to identify which Perfect Clear solves are more human-like—especially which solutions people are more likely to find intuitive and choose in practice.

Solve-choice model. Where v3 predicted *placements step by step*, v4 scores a
**finished solve** and outputs one number: the probability that the player
would have picked this solve, played in this order.

## Input

Each sample is the 4x10 **broken piece board** (the board where cleared lines
remain in place — same object as PCReview v3's `completed` rows, but built
exactly from the solve's placements, so all 40 cells are filled for a 4L PC)
plus the **piece ordering**:

- `color[40]` — per cell: empty or one of the 7 piece colors.
  Embedded in 8 dims by `MirrorColorEmbedding`, which *structurally enforces*
  L/J and S/Z symmetry (L = b + d, J = b − d; likewise S/Z), so mirrored
  pieces stay exact reflections in embedding space for all of training.
- `ordinal[40]` — per cell: 0 (empty) or 1–10, the ordinal of the piece that
  filled the cell in this play order. Embedded in 8 dims.

Total: 40 x 8 (color) + 40 x 8 (ordinal), concatenated per cell to a
16-channel 4x10 grid.

## Model (`pcnn4/model.py`)

`PCChoiceNet`: embeddings → 3x Conv2d (64/128/128, 3x3) → flatten → MLP
(256, 64) → **1 logit** (sigmoid = choice probability). ~1.5M params.
Trained with weighted `BCEWithLogitsLoss`.

## Training data (`pcnn4/build_dataset.py`)

For every full 4L PC in the v3 human corpus (`PC-NN-v3/pc-nn/data/parsed_weighted`):

1. Reconstruct the 11-piece stream and true PC number (v3 `build_weights`).
2. Enumerate **all** solves of the stream on the empty board with sfinder
   (`path`, hold use), cached per stream in `data/sfinder_cache/`.
3. Compute each solve's save → next-segment V* (v3 `pc_value`).
4. Identify the player's chosen solve by exact placement-set match in
   original coordinates (replayed from the parsed steps' `delta`s).
5. Let `x` = chosen V*, `d = V*(solve) − x`. Keep solves with `d > −1`
   (`v_window`), i.e. everything not clearly worse than the choice.
6. Expand every kept solve into **all playable piece orders**
   (v3 `solution_order.enumerate_playable_orders`, reachability-validated),
   including all orders of the chosen solve. One sample per (solve, order).

### Labels and weights (`pcnn4/weighting.py`)

- Chosen solve's orders → `label 1`; all other solves' orders → `label 0`.
- Negative weight = `gate(d) * boost(d)`:
  - `gate`: smoothstep 0→1 over `d ∈ [−1, 0]` — slightly-worse saves count,
    with fading weight;
  - `boost`: `10 ** smoothstep(d / 0.5)` — an alternative with equal V*
    weighs 1x, one the player skipped despite a significantly better save
    (d ≥ 0.5) weighs **10x**.
- Positive samples share `pos_ratio ×` the situation's total negative weight
  (floor 1.0), so the lone chosen solve isn't drowned by its many negatives.
- `order_weight_mode: 'split'` divides a solve's weight across its orders
  (set `'replicate'` in `config.py` to give every order the full weight).

Caveat worth knowing: real V* spreads inside one situation are often <0.01
on a ~3842–4353 scale, so most negatives sit near weight 1 and the 10x boost
fires only on genuinely skipped-value choices. Tune `d_significant` /
`v_window` in `WEIGHT_CONFIG` if that proves too blunt.

### Augmentation

50% horizontal mirror (flip rows, swap L↔J / S↔Z) at dataset level —
consistent with the enforced embedding symmetry. Note mirroring does not
remap the V*-derived weight (the mirrored stream's V* can differ); it is
treated as pure augmentation, same as v3.

## Layout

```
pcnn4/
  config.py         constants, model/weight/train config, paths
  v3bridge.py       puts PC-NN-v3/pc-nn on sys.path (reuses its pipeline)
  embeddings.py     MirrorColorEmbedding (enforced L/J, S/Z symmetry), OrdinalEmbedding
  model.py          PCChoiceNet -> single logit
  encoding.py       placements+order -> grids; replay steps -> original placements
  weighting.py      V* gate/boost, positive balancing
  build_dataset.py  corpus -> data/dataset.jsonl (one line per solve-order)
  dataset.py        JSONL Dataset, situation-level train/val split, mirror aug
  train.py          weighted-BCE training loop (NOT run yet)
tests/              pytest suite (unit + real-data integration; no sfinder)
```

## Commands

Python: `/usr/bin/python3` (3.9, has torch 2.8 / py_fumen_py / pytest).

```bash
# tests
cd "/Users/jesse/Documents/PCMode/PC-NN-v4" && /usr/bin/python3 -m pytest tests/ -q

# build dataset (sfinder required; start small)
/usr/bin/python3 -m pcnn4.build_dataset --limit-files 2

# train (later)
/usr/bin/python3 -m pcnn4.train
```
