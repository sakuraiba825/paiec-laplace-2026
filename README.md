# A Simple Beta-Binomial Baseline for Predictive AI Evaluation with Sparse Acquired Labels

**Technical report accepted at PAIEC 2026, an official competition in the NeurIPS 2026 Competition Track.**

- **Author:** Yutong Xie (Tencent)
- **OpenReview:** [Submission #10](https://openreview.net/forum?id=oA7JnviL2M)
- **Competition:** [Predictive AI Evaluation Competition at NeurIPS 2026](https://aimslab.stanford.edu/competition)
- **Final Codabench submission ID:** `1004923`
- **Code license:** MIT
- **Technical report license:** CC BY 4.0

This acceptance applies to the **PAIEC competition technical report**. It is not a NeurIPS 2026 main-track paper acceptance and does not, by itself, establish an oral/poster assignment or a final competition ranking.

## Overview

The Predictive AI Evaluation Competition (PAIEC) asks participants to estimate the probability that an AI system will respond successfully to a benchmark item without running that system on the item. Predictions may use the supplied subject and item attributes together with a small number of acquired labels.

This repository contains the frozen final predictor, a Jeffreys-prior comparator, synthetic tests, and a reproducible public-development diagnostic.

The final predictor is a conventional Laplace Beta(1,1) posterior mean. For `n` matching acquired labels containing `s` successes, it returns:

```text
p = (1 + s) / (2 + n)
```

Labels match only when all eight subject attributes and the `benchmark_id` agree with the target. With no matching labels, the prediction is `0.5`.

The runtime model:

- uses only Python's standard library;
- performs no training, API calls, model downloads, or network access;
- uses no fitted public-data prior or LLM inference;
- includes no acquisition-function override; and
- is byte-identical to the `model.py` in final Codabench submission `1004923`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `model.py` | Frozen Laplace Beta(1,1) runtime predictor |
| `jeffreys_reference.py` | Jeffreys Beta(0.5,0.5) public comparator |
| `reproduce_public.py` | Reproduction entry point for the public 74-domain diagnostic |
| `public_manifest.json` | Frozen public data revision, selected pairs, files, sizes, and hashes |
| `public_reference_summary.json` | Reference numerical results for reproduction checks |
| `reproduction/public_summary.json` | Completed release-validation output |
| `core/` | Public-data loading and evaluation protocol modules |
| `tests/` | Synthetic model, protocol, and reproduction tests |
| `release_integrity.json` | Release file hashes and frozen artifact identity |
| `RUN_INSTRUCTIONS.txt` | Detailed data, protocol, validation, and disclosure notes |

## Quick start

The runtime predictor itself has no third-party dependencies. The pinned packages below are needed for the full synthetic test suite and public-data reproduction.

```bash
git clone https://github.com/sakuraiba825/paiec-laplace-2026.git
cd paiec-laplace-2026
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s tests -v
```

Validated analysis environment: CPython 3.12.13, Polars 2.1.0, and PyArrow 26.0.0.

## Public-data reproduction

Obtain [`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db) through its original access requirements and pin revision:

```text
54074453da7b9911be0f48e4c560be547bc84d21
```

Download only the 296 Parquet tables listed in `public_manifest.json`. The data directory must directly contain paths such as:

```text
afrieval/formatted_tables/benchmarks.parquet
afrieval/formatted_tables/subjects.parquet
afrieval/formatted_tables/items.parquet
afrieval/formatted_tables/responses.parquet
```

Then run:

```bash
.venv/bin/python -B reproduce_public.py \
  --data-dir /path/to/public_snapshot \
  --output-dir /path/to/new_public_output
```

The output directory must not already exist. See [`RUN_INSTRUCTIONS.txt`](RUN_INSTRUCTIONS.txt) for the complete layout, protocol, and output description.

## Recorded release validation

- 29 synthetic tests passed.
- 74 public development domains with three seeds each: 222 domain-seed results.
- 296 public source tables verified by size and SHA-256.
- 4,017 numerical values checked against the reference summary.
- Maximum absolute reference difference: `1.6653345369377348e-16`.
- Laplace public-panel ALC: `0.22503754069056203`.
- Jeffreys public-panel ALC: `0.23094111247206206`.

These are **public-development diagnostics, not a new blind holdout and not official private-test results**. The 74 domains were inspected during development, and the three seeds do not constitute 222 independent domains. Local splitting, response selection, NULL handling, and acquisition ordering are explicit simulation assumptions rather than a byte-for-byte reproduction of the hidden evaluation coordinator.

The release does not redistribute source Parquet files, question text, response labels, private-test records, credentials, or private-feedback-derived artifacts. Original dataset licenses and access terms remain applicable.

## Citation

```bibtex
@misc{xie2026paiec,
  author       = {Xie, Yutong},
  title        = {A Simple Beta-Binomial Baseline for Predictive AI Evaluation with Sparse Acquired Labels},
  year         = {2026},
  howpublished = {\url{https://openreview.net/forum?id=oA7JnviL2M}},
  note         = {Technical report accepted at PAIEC 2026, NeurIPS 2026 Competition Track}
}
```

## License

The original code is released under the [MIT License](LICENSE.txt). The technical report is licensed under CC BY 4.0. These licenses do not change the terms of third-party materials or source datasets.
