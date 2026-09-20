# Running the code with a separately downloaded dataset

All examples start in the repository root and use a Linux shell. Use Python 3.10
and separate task environments. Set `PKUEEG_DATA` to the root containing
`dataset_description.json`, `sub-01/`, `stimuli/`, and `derivatives/`.
Set `PKUEEG_OUT` to a new writable directory outside both the data and code.
Do not rely on the preserved package-relative defaults in a code-only checkout.

## Analysis environment

```bash
python -m venv ../pkueeg-analysis-env
source ../pkueeg-analysis-env/bin/activate
python -m pip install -r requirements/analysis.txt
export PKUEEG_DATA="/path/to/downloaded/PKUEEG"
export PKUEEG_OUT="/path/to/new/PKUEEG-results"
mkdir -p "$PKUEEG_OUT"
```

The supplied requirement lists describe direct dependencies, not a complete
cross-platform lock. Record the resolved environment for each run. Feature
extraction requires additional packages/models: see [FEATURES.md](FEATURES.md).
The original root `code/requirements.txt` is intentionally preserved as provenance,
not presented as a sufficient environment for every task.

## Retrieval: input check, training and plotting

```bash
python -B scripts/prepare_retrieval_config.py \
  --dataset-root "$PKUEEG_DATA" \
  --output-config "$PKUEEG_OUT/retrieval.yaml" \
  --output-dir "$PKUEEG_OUT/retrieval-run"

python -B code/speech_retrieval/run_experiment.py \
  --config "$PKUEEG_OUT/retrieval.yaml" --validate-only \
  --output-dir "$PKUEEG_OUT/retrieval-input-check"

# This is real training, not a data-free smoke test.
python -B code/speech_retrieval/run_experiment.py \
  --config "$PKUEEG_OUT/retrieval.yaml" --subjects sub-01 \
  --output-dir "$PKUEEG_OUT/retrieval-sub01"

```

### Explicit optional full run

The following block trains all 25 participants. Run it only when a full experiment
is intended; it is not required for an input check or publication preflight.

```bash
# Full experiment; it can be computationally expensive.
python -B code/speech_retrieval/run_experiment.py \
  --config "$PKUEEG_OUT/retrieval.yaml"

python -B code/speech_retrieval/analyze_results.py \
  --output-dir "$PKUEEG_OUT/retrieval-run"
```

The helper rewrites only paths and explicitly selected features, preserving the
original scientific hyperparameters. Config paths are relative to the generated
YAML file. On Windows, code/data/config/result paths used by retrieval should be
on the same drive; path snapshots use relative paths.

Defaults remain envelope plus wav2vec, with subject/day-specific story splits,
13/12/13 training stories, two validation and two test stories, 3/5/10-second
windows, five candidates, and flattened Pearson matching. Group figures use
SEM. Do not confuse these with manuscript figures using different aggregation
or confidence-interval conventions. The stored 50-Hz wav2vec input is resampled
to the 100-Hz analysis grid in memory; it is not native 100-Hz model output.

For an explicitly authorized dataset/derivative containing all 50 static-word
arrays, make a **different** config and result directory:

```bash
python -B scripts/prepare_retrieval_config.py \
  --dataset-root "$PKUEEG_DATA" \
  --output-config "$PKUEEG_OUT/retrieval-three-features.yaml" \
  --output-dir "$PKUEEG_OUT/retrieval-three-features" \
  --features envelope wav2vec word2vec
```

This does not generate missing arrays, verify their scientific equivalence,
change the downloaded data, or update an OpenNeuro snapshot.

## Completeness and timing

```bash
python -B code/technical_validation/pkueeg_completeness_temporal.py \
  --eeg-dir "$PKUEEG_DATA/derivatives/preproc_40hz" \
  --envelope-dir "$PKUEEG_DATA/derivatives/stimulus_features/envelope/envelope_100hz" \
  --output-dir "$PKUEEG_OUT/completeness"
```

This reads real EEG arrays across the dataset. It is not merely a fast filename
check and is not part of the data-free test suite.

## Forward speech-envelope prediction

```bash
python -B code/technical_validation/pkueeg_trf_prediction.py \
  --formal-dir "$PKUEEG_DATA/derivatives/preproc_40hz" \
  --envelope-dir "$PKUEEG_DATA/derivatives/stimulus_features/envelope/envelope_100hz" \
  --output-dir "$PKUEEG_OUT/prediction-sub01-day1" \
  --subjects sub-01 --days day1 --n-jobs 1
```

This fits the full Day-1 story set for one subject. The imported protocol uses
-300 to 600 ms delays, 250-Hz analysis, and nested story-disjoint cross-validation.
Remove the subject/day restrictions for the full analysis and choose a new output
directory. Synthetic `--self-test` does not use real recordings.

## Listening/rest ISC and descriptive TRF waveforms

```bash
python -B code/technical_validation/pkueeg_validation_recheck.py \
  --formal-dir "$PKUEEG_DATA/derivatives/preproc_40hz" \
  --envelope-dir "$PKUEEG_DATA/derivatives/stimulus_features/envelope/envelope_100hz" \
  --results-dir "$PKUEEG_OUT/isc-trf" --analyses isc,trf --n-jobs 1
```

Consult `--help` and the task README for statistical definitions. Descriptive
TRF waveform fits and held-out EEG prediction are different analyses.

## Optional behavior-neural analysis

An authorized external long-format TSV is required: `participant_id`,
`session_id` (`ses-day1` through `ses-day3`), and `comprehension_accuracy` in
[0, 1], with one row for each of 25 subjects x 3 days. This table is not supplied
or synthesized. First obtain new prediction scores from the matching data.

```bash
python -B code/technical_validation/pkueeg_behavior_analysis.py \
  --behavior /path/to/authorized_behavior_long.tsv \
  --neural-summary /path/to/prediction/subject_day_summary.csv \
  --output-dir "$PKUEEG_OUT/behavior"
```

## Legacy EEG preprocessing

```bash
python -B code/preprocessing/preprocess_pkueeg.py \
  --bids-root "$PKUEEG_DATA" --output-root "$PKUEEG_OUT/preprocessing" \
  --participants sub-01 --sessions day1 --n-jobs 1
```

The outputs are separate `preproc_40hz` (250 Hz) and `preproc_8hz` (128 Hz)
directories. This is the preserved legacy implementation: it has case-sensitive
montage/bad-channel handling, does not explicitly remove bad time segments, and
does not save per-session ICA objects. Do not describe it as the later repair
pipeline or claim it reproduces every published sample exactly. MNE reads raw
BrainVision data in volts; the script saves EEG arrays in microvolts. Do not
apply another historical Day-3 scale correction to already unified derivatives.

## Raw quality checks and structural validation

For raw QC, copy `code/technical_validation/data_quality/config.yaml` to an
external workspace and set its `raw_bids_dir`, `preprocessed_dir`,
`stimulus_audio_dir`, `stimulus_envelope_dir`, and `output_dir` under `paths`
to the intended data and a fresh external output.
Paths resolve relative to that YAML file. Consult the actual configuration keys
before editing. Then run:

```bash
python -B code/technical_validation/data_quality/code/analyze_raw.py \
  --config /path/to/external/quality.yaml --subjects sub-01
python -B code/technical_validation/data_quality/code/aggregate_quality.py \
  --config /path/to/external/quality.yaml

python -B code/validate_layout.py --root "$PKUEEG_DATA" \
  --report "$PKUEEG_OUT/layout.json"
python -B code/run_bids_validation.py --root "$PKUEEG_DATA" \
  --output-dir "$PKUEEG_OUT/bids" --validator bids-validator-deno
```

The BIDS validator is an external prerequisite; this repository does not install
it. Layout/BIDS checks, signal-quality summaries, and scientific validation have
different scopes. The original data package can retain historical status files;
do not take a status-only folder as completed processing evidence.
