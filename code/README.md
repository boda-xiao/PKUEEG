# PKUEEG code modules

This directory preserves the original release code layout and scientific Python
implementations. For the standalone GitHub checkout, start with the
[repository README](../README.md), [explicit-path commands](../docs/RUNNING.md),
and [feature guide](../docs/FEATURES.md). Do not assume data exist beside this code.

- `preprocessing/preprocess_pkueeg.py`: legacy EEG processing; `config/` is beside it.
- `feature_extraction/stimulus_features/`: envelope, Mel, Chinese wav2vec2,
  frozen fastText/static-word extraction, BERT v3, and a retained legacy BERT script.
- `technical_validation/`: completeness/timing, ISC, envelope TRF and held-out
  prediction, signal QC, and optional external-table behavioral analysis.
- `speech_retrieval/`: lagged multi-output ridge decoding and five-candidate matching.
- `validate_layout.py`: read-only structure, metadata and array-header checks.
- `run_bids_validation.py`: external BIDS-validator runner, not scientific validation.
- `model_downloads/`: download-integration status; neural weights are not bundled.
- `provenance/legacy_layout/`: historical builders/verifiers, not current entry points.

`config/release_scope.json` records the earlier 2026-09-19 integration state. Its
text about absent word2vec describes that historical state, not a current inventory
of the server package or this code-only checkout. The later local package contains
50 word2vec arrays; the GitHub repository does not include those time-series arrays.
Neither the local additions nor this code upload change OpenNeuro 1.0.0.

## Environments and input/output safety

The retained `requirements.txt` contains only MNE, NumPy and SciPy. For analyses
and preprocessing, use `../requirements/analysis.txt`, which also includes
scikit-learn for the default FastICA path. Feature recipes have separate direct
dependency lists and model fingerprints; these are not complete environment locks.
The legacy BERT script does not share the new BERT-v3 model/windowing contract.

All data paths must be explicit for a standalone checkout. Original shell wrappers
assume the embedded dataset layout; use the commands in `../docs/RUNNING.md`
instead. Keep code, downloaded data, and fresh output directories separate and
mutually non-overlapping. Do not rely on legacy guards to protect every sibling
directory of an externally supplied dataset.

## Scientific and resource limitations

The legacy EEG entry point has case-sensitive montage/bad-channel matching,
no explicit bad-segment rejection, and no saved per-session ICA records. New
feature recipes do not guarantee exact reproduction of stored legacy arrays.
The BERT time rasterizer needs external validated MFA-v3 outputs; an executable
MFA workflow is not bundled. Model files and behavioral accuracy are external.
Scientific results are not distributed, and no full scientific rerun is claimed
for this repository publication.

The default retrieval configuration remains envelope plus wav2vec. Use the new
external-config helper to select word2vec only when the actual required inputs
are available. Stored 50-Hz wav2vec is resampled to the analysis grid in memory;
it is not native 100-Hz extraction.
