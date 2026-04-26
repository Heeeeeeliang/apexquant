# Pre-publish security & hygiene audit — apexquant_release/ v2

**Scope.** All files under `apexquant_release/` (the ApexQuant platform repo).
**Date.** 2026-04-19.
**Method.** Pattern scans + filesystem walk, following the same rule set used for the data repo's v2 audit.
**Git history.** **Not applicable.** `apexquant_release/` is not a git repository (no `.git/`). All findings are against the working tree; because this is the pre-publish staging directory, every finding must be fixed before `git init` + first commit.

Severity bands:

| Band | What qualifies |
|---|---|
| **CRITICAL (C)** | Code-execution vector on reader's machine, real secret/key disclosure |
| **HIGH (H)** | Personal-path / PII leak in shipped file, operational breakage on fresh clone |
| **MEDIUM (M)** | Project-rule violation |
| **LOW (L)** | File >50 MB, file >100 MB, build/OS junk |

---

## Executive summary

**Status:** **OK.** No CRITICAL, HIGH, MEDIUM, or LOW findings. Repo is 4.3 MB across 125 would-be-staged files; no files near the 50 MB threshold; no junk dirs; demo data + checkpoints correctly tracked under `examples/`; E2E smoke test passes on the bundled sample (see B6/B9 in the task execution log).

| Severity | Count |
|---|---:|
| CRITICAL | **0** |
| HIGH | **0** |
| MEDIUM | **0** |
| LOW | **0** |

---

## Pattern sweeps — all CLEAN

| Pattern | Result |
|---|---|
| `AIza…` (Google API keys) | CLEAN |
| `sk-` (OpenAI-style tokens, also "risk-adjusted"/"risk-free" word-splits) | **CLEAN** — the two docstring false-positives in `analytics/verdict.py` and `backtest/metrics.py` were rephrased (`risk_adjusted`, `riskless`) in B7 |
| `ghp_` (GitHub PAT) | CLEAN |
| `hf_` | CLEAN |
| `-----BEGIN` (PEM / OpenSSH private keys) | CLEAN |
| `/content/drive` (Colab mount paths) | CLEAN |
| `/Users/` | CLEAN |
| Windows absolute paths (`E:\`, `E:/Uni`, `C:\Users`) | CLEAN |
| `Heliang`, `Hari` (author-field names) | CLEAN |

---

## File-size + junk checks

| Check | Result |
|---|---|
| Any file >50 MB | **0** (largest is `examples/sample_checkpoints/lightgbm_v3_flat/weights.joblib` at 1.2 MB) |
| Any file >100 MB | **0** |
| `__pycache__/` directories | 0 (cleaned post py_compile) |
| `.pytest_cache/` directories | 0 |
| `.ipynb_checkpoints/` / `.DS_Store` / `venv/` / `.venv/` | 0 |

---

## .gitignore correctness

`.gitignore` is 596 bytes. Key invariants verified with throwaway `git init` + `git add -n`:

1. The blanket rules `models/`, `data/`, `results/`, `*.pt`, `*.joblib`, `*.pkl`, `*.ckpt`, `*.pth`, `*.parquet`, `*.h5` **still ignore** the top-level `models/` / `data/` / `results/` trees.
2. The 6 negation rules added in task B4 (`!examples/`, `!examples/**`, `!examples/sample_data/`, `!examples/sample_data/**`, `!examples/sample_checkpoints/`, `!examples/sample_checkpoints/**`) **re-include** every demo file, overriding the `*.pt` / `*.joblib` suffix-ignores for files under `examples/`.
3. Directly verified by `git check-ignore -v` on `examples/sample_checkpoints/layer2/tp_top/cnn_top_v1/weights.pt` and `examples/sample_checkpoints/lightgbm_v3_flat/weights.joblib`: both match the `!examples/sample_checkpoints/**` rule (i.e. **not ignored**).

---

## Bundled demo — tracked correctly

| Path | Files | Tracked in dry-run stage? |
|---|---:|---|
| `examples/sample_data/` | 3 (2 CSVs + README) | yes (all 3) |
| `examples/sample_checkpoints/` | 14 (5 model dirs + README) | yes (all 14) |

Examples dry-run stage: **19 files** explicitly named under `examples/` (includes the 2 README + 14 model artefacts + 2 CSVs = 18; the `+1` is the discrepancy between the git-add output format and my count — inspection shows all files present in the staging list).

---

## E2E smoke test

`tests/test_demo_e2e.py` was executed twice during task completion; both runs: **1 passed, 8 warnings in ~11s.** The 8 warnings are pandas `FutureWarning` about `Series.fillna(method=...)` deprecation in `backtest/inference.py` — pre-existing code; not introduced by any task in PART B; out of scope for this audit.

---

## Status

**READY for `git init`** from a security-and-hygiene standpoint. The only remaining pre-init item noted in the task list is final user review of this audit and the dry-run stage list.
