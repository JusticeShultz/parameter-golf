# Parameter Golf Research Tracks

Priority order is dictated by the challenge rules:

1. stay under the `16,000,000` byte artifact cap
2. stay within the `10 minute / 8xH100` training budget for record attempts
3. optimize post-roundtrip `val_bpb`, not pre-quant loss

## Integrated now

- Post-compression-aware training:
  - sampled int8 reconstruction regularizer
  - optional ternary-weight regularizer
  - optional outlier suppression penalty
- Weight sharing / recurrence:
  - shared-block transformer via `NUM_UNIQUE_BLOCKS`
- Sparse attention:
  - optional sliding-window attention via `WINDOW_SIZE`
- Factorized embeddings:
  - optional `EMBED_DIM < MODEL_DIM`
- Hybrid eval-time compute:
  - optional recent-token cache bias during validation / roundtrip eval
- Local proxy iteration:
  - capped validation
  - optional skip of expensive final roundtrip eval
  - proxy sweep launcher

## Current knobs

- `NUM_UNIQUE_BLOCKS`
- `WINDOW_SIZE`
- `EMBED_DIM`
- `COMPRESSION_REG_WEIGHT`
- `TERNARY_REG_WEIGHT`
- `OUTLIER_REG_WEIGHT`
- `EVAL_CACHE_MIX_WEIGHT`
- `EVAL_BIGRAM_MIX_WEIGHT`
- `EVAL_CACHE_SIZE`
- `FINAL_ROUNDTRIP_EVAL`
- `ROUNDTRIP_VAL_MAX_TOKENS`

## Local proxy reference point

All local comparisons below use the same quick 3090 proxy envelope:

- `MAX_WALLCLOCK_SECONDS=180`
- `TRAIN_BATCH_TOKENS=32768`
- `VAL_MAX_TOKENS=1048576`
- `FINAL_ROUNDTRIP_EVAL=0`
- baseline architecture:
  - `NUM_LAYERS=12`
  - `NUM_UNIQUE_BLOCKS=12`
  - `MODEL_DIM=384`
  - `EMBED_DIM=0`
  - `NUM_HEADS=6`
  - `NUM_KV_HEADS=3`

## Roundtrip proxy track

Use this when ranking experiments on a more faithful local objective:

- keep the same baseline architecture unless explicitly testing architecture
- enable `FINAL_ROUNDTRIP_EVAL=1`
- keep `ROUNDTRIP_VAL_MAX_TOKENS` capped so the run stays practical on a 3090
- treat this as the local approximation to the actual challenge metric

## Latest findings

- Quick local baseline:
  - run: `baseline3090_20260318_170251`
  - result: `val_bpb=2.0916`, `val_loss=3.4910`
  - total artifact: `6,831,983` bytes
  - interpretation: current local number to beat
- Hybrid eval sidecar, recent-token + bigram continuation bias:
  - run: `sidecar3090_20260318_172524`
  - knobs: `EVAL_CACHE_MIX_WEIGHT=0.03`, `EVAL_BIGRAM_MIX_WEIGHT=0.05`, `EVAL_CACHE_SIZE=16`
  - result: `val_bpb=2.0970`, `val_loss=3.5000`
  - total artifact: `6,810,819` bytes
  - delta vs baseline: `+0.0054 bpb` worse, `21,164` bytes smaller
  - interpretation: close enough to keep around for later tuning, not good enough to become the default path
- Compression-aware baseline, reconstruction regularization `0.01`:
  - run: `compress3090_20260318_174132`
  - result: `val_bpb=2.0943`, `val_loss=3.4954`
  - total artifact: `6,812,935` bytes
  - delta vs baseline: `+0.0027 bpb` worse, `19,048` bytes smaller
  - interpretation: strongest experimental branch so far
- Compression-aware baseline, reconstruction regularization `0.005`:
  - run: `compress3090_half_20260318_1750`
  - result: `val_bpb=2.0928`, `val_loss=3.4930`
  - total artifact: `6,829,073` bytes
  - delta vs baseline: `+0.0012 bpb` worse, `2,910` bytes smaller
  - interpretation: best pre-roundtrip proxy result outside the plain baseline
- Matched roundtrip-proxy baseline:
  - run: `baselinert3090_20260318_181344`
  - exact final roundtrip result: `val_bpb=2.11089617`, `val_loss=3.56464830`
  - total artifact: `6,705,058` bytes
- Matched roundtrip-proxy compression baseline:
  - run: `compressrt3090_20260318_175828`
  - knobs: `COMPRESSION_REG_WEIGHT=0.005`
  - exact final roundtrip result: `val_bpb=2.06085837`, `val_loss=3.48014999`
  - total artifact: `6,839,798` bytes
  - delta vs matched roundtrip baseline: `-0.05003780 bpb`, about `2.37%` better
  - interpretation: compression-aware training is now the leading local research branch when measured on a more faithful objective
- Sparse-attention probe on the winning compression setup:
  - run: `compressrt_sparse512_20260318_1842`
  - knobs: `WINDOW_SIZE=512`, `COMPRESSION_REG_WEIGHT=0.005`
  - exact final roundtrip result: `val_bpb=2.07004634`, `val_loss=3.49566562`
  - delta vs best compression baseline: `+0.00918797 bpb` worse
  - interpretation: not good enough to displace the dense compression-aware path; sparse attention stays experimental for later
- Focused QAT roundtrip sweep around the winning compression point:
  - sweep: `qatrtsweep_20260318_1906`
  - best result in sweep:
    - run: `qatrtsweep_20260318_1906_w0045_o0000`
    - knobs: `COMPRESSION_REG_WEIGHT=0.0045`, `OUTLIER_REG_WEIGHT=0.0`
    - exact final roundtrip result: `val_bpb=2.06804196`, `val_loss=3.49228084`
    - total artifact: `6,814,995` bytes
  - interpretation:
    - tiny outlier regularization did not help on this local roundtrip track
    - none of the focused QAT sweep runs beat the standing best dense compression-aware run at `2.06085837`
    - the dense compression-aware baseline remains the current best local result
- Recurrent/shared-block roundtrip sweep:
  - sweep: `recurtsweep_20260318_1925`
  - tested:
    - `16 layers / 8 unique / embed 0` -> `2.25452146`
    - `18 layers / 6 unique / embed 0` -> `2.28804085`
    - `16 layers / 8 unique / embed 256` -> `2.28260194`
    - `18 layers / 6 unique / embed 256` -> `2.34886036`
  - interpretation:
    - this branch cuts artifact size aggressively, but quality collapses on the current local roundtrip track
    - none of these shapes are close to the dense compression-aware baseline
    - shared-block recurrence stays interesting for the 16 MB objective, but this first pass is not competitive enough to prioritize locally
- Roundtrip sidecar revisit on top of the winning dense compression setup:
  - sweep: `sidecarrtsweep_20260318_1942`
  - best usable result in sweep:
    - run: `sidecarrtsweep_20260318_1942_c0020_b0030_s8`
    - knobs: `EVAL_CACHE_MIX_WEIGHT=0.02`, `EVAL_BIGRAM_MIX_WEIGHT=0.03`, `EVAL_CACHE_SIZE=8`
    - exact final roundtrip result: `val_bpb=2.06132482`, `val_loss=3.48093767`
    - total artifact: `6,864,315` bytes
    - delta vs best dense compression baseline: `+0.00046645 bpb` worse
  - sweep reliability notes:
    - `c0015_b0020_s8` and `c0020_b0020_s8` stopped before a usable roundtrip result was written
    - `c0020_b0020_s16` reached artifact export but never wrote `final_int8_zlib_roundtrip_exact`
  - interpretation:
    - the sidecar branch is the closest secondary idea so far
    - it still did not beat the plain dense compression-aware winner
    - keep it parked as a late-stage add-on, not the current pivot

## Current leader

- `compressrt3090_20260318_175828`
- dense attention, no sidecar, no recurrence, no factorized embedding
- `COMPRESSION_REG_WEIGHT=0.005`
- exact final roundtrip result: `val_bpb=2.06085837`
- total artifact: `6,839,798` bytes

## Immediate next step

- Pivot into native low-bit shaping on the roundtrip track
- keep the dense compression-aware baseline fixed
- add only very small `TERNARY_REG_WEIGHT` values first
- keep `OUTLIER_REG_WEIGHT=0` until ternary-only behavior is measured cleanly
- rank experiments by `final_int8_zlib_roundtrip_exact val_bpb`

## Next experiments

- Native low-bit / ternary shaping:
  - sweep conservative `TERNARY_REG_WEIGHT` values on top of the winning dense compression-aware setup
  - keep architecture and eval settings fixed
  - decide first whether ternary pressure helps the actual roundtripped metric at all
- Compression + ternary interaction:
  - if a ternary value helps, retune `COMPRESSION_REG_WEIGHT` around that point
  - only then reconsider tiny outlier suppression
- Sidecar follow-up, only if low-bit shaping stalls:
  - rerun the closest sidecar point on a clean single-run path
  - only continue if it can beat the dense no-sidecar baseline
  - otherwise keep the sidecar idea as a later ensemble/mixer branch

## Medium-term work

- Dense winner + sidecar + low-bit combined into one trainer once the individual branches are measured cleanly
- Global/shared codebook quantization across layers
- Basis-generated per-layer weights or hypernetwork-style weight generation
- Test-time adaptation with strict reset semantics
- Token-adaptive recurrent depth / halting policy

## Deferred until the model is stronger

- Tokenizer redesign
- aggressive code-size golf
- heavy hyperparameter brute force
