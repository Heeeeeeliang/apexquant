# Bundled sample checkpoints — demo only

This directory carries the minimal set of trained model weights needed to
run `tests/test_demo_e2e.py` and the Streamlit UI's "Run AI Backtest"
button end-to-end on the bundled sample data with the `run9_trailstop`
preset. It is **not** the full model zoo — those live on HuggingFace; see
the research repo.

## Layout and role

The subfolder structure matches what
`predictors._discover_and_register(models_dir=...)` expects. Each model
directory carries a `meta.json` (adapter type, output_label) plus its
weights file; LightGBM directories additionally carry
`feature_names.json` for name-based feature alignment.

| Path (relative to `sample_checkpoints/`) | Adapter | Output label | Role in the run9 pipeline | Size |
|---|---|---|---|---:|
| `lightgbm_v3_flat/weights.joblib` | `VolAdapter` (regressor) | `vol_prob_flat` | Layer 1 — volatility regime forecast; drives the Vol Gate | 1.2 MB |
| `layer2/tp_top/cnn_top_v1/weights.pt` | `CnnAdapter` | `tp_top_prob` | Layer 2 — local top (sell-signal) detector | 227 KB |
| `layer2/tp_bottom/cnn_bottom_v1/weights.pt` | `CnnAdapter` | `tp_bottom_prob` | Layer 2 — local bottom (buy-signal) detector | 227 KB |
| `lgb_top_v1/weights.joblib` | `MetaAdapter` (classifier) | `tp_top` | Layer 3 — top-signal meta-label trade filter | 285 KB |
| `lgb_bottom_v1/weights.joblib` | `MetaAdapter` (classifier) | `tp_bottom` | Layer 3 — bottom-signal meta-label trade filter | 414 KB |

Total: 5 model directories, 13 files, ~2.4 MB.

## How the pipeline picks them up

- `backtest.inference.generate_predictions(config)` calls
  `REGISTRY.reload(models_dir)` which rglobs for `meta.json`, reads the
  `adapter` field, instantiates the right adapter class, and calls
  `.load()` to deserialise the weights file.
- The built-in `AIStrategy` (used by `run9_trailstop`) reads the output
  labels via fuzzy substring match: `"vol"` → `vol_prob_flat`,
  `"top"`/`"bottom"` → the CNN probabilities (or meta-label classifiers
  if fuzzy match picks them first).
- To point the discovery at this bundle rather than the repo's
  top-level `models/`, pass `models_dir="examples/sample_checkpoints"`
  to `REGISTRY.reload`. The E2E test does this automatically.

## Loading safely

All `torch.load(...)` calls in this repo use `weights_only=True` by
default — see the security audit. The `.pt` checkpoints here load under
that safe path; no pickle-based object deserialisation is required for
these weights.
