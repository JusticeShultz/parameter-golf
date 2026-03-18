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
- `EMBED_DIM`
- `COMPRESSION_REG_WEIGHT`
- `TERNARY_REG_WEIGHT`
- `OUTLIER_REG_WEIGHT`
- `EVAL_CACHE_MIX_WEIGHT`
- `EVAL_CACHE_SIZE`
- `FINAL_ROUNDTRIP_EVAL`
- `ROUNDTRIP_VAL_MAX_TOKENS`

## Next experiments

- Zlib-aware QAT baseline:
  - sweep compression / ternary / outlier penalties
  - rank by proxy post-compression metrics
- Recurrent shared-block transformer:
  - vary `NUM_LAYERS`, `NUM_UNIQUE_BLOCKS`, `EMBED_DIM`
  - test whether smaller unique depth plus more effective depth improves proxy `val_bpb`
- Tiny hybrid sidecar model:
  - replace the current recency bias with a real adaptive mixer over:
    - recent-token cache
    - tiny n-gram model
    - neural logits

## Medium-term work

- Global/shared codebook quantization across layers
- Basis-generated per-layer weights or hypernetwork-style weight generation
- Test-time adaptation with strict reset semantics
- Token-adaptive recurrent depth / halting policy

## Deferred until the model is stronger

- Tokenizer redesign
- aggressive code-size golf
- heavy hyperparameter brute force
