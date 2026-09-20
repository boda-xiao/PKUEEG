# PKUEEG speech retrieval: public-release integration

For a standalone GitHub checkout with data stored separately, first use
`scripts/prepare_retrieval_config.py` from the repository root; see
[RUNNING.md](../../docs/RUNNING.md). The embedded-layout default commands below
require code beside the dataset and are not the standalone Quickstart.

This is the speech-retrieval implementation copied from the 20260908 candidate
and adapted on 2026-09-19 to `PKUEEG_release_20260909`. It reconstructs speech
features using a lagged multi-output ridge decoder, then performs five-candidate
segment retrieval. No old results, fitted models, or feature arrays are included.

**This is a code migration, not a reproduction of historical results.** The
current public EEG and features differ from the candidate/repair inputs. The
default preserves the earlier two-feature configuration rather than
automatically switching to the later local three-feature set.
No full 25-subject training rerun was performed during this integration.

## Inputs and alignment

All configured input paths are relative to the YAML file, not the working directory.

| Input | Release-root-relative location | Native sampling rate |
|---|---|---:|
| EEG | `derivatives/preproc_40hz/sub-XX/ses-dayD/eeg/sub-XX_ses-dayD_task-audio_desc-storyNN_eeg.npz` | 250 Hz |
| Envelope | `derivatives/stimulus_features/envelope/envelope_100hz/N_envelope.npy` | 100 Hz |
| wav2vec | `derivatives/stimulus_features/wav2vec2/wav2vec2_layer9_50hz/story-NN_wav2vec2-layer9.npy` | 50 Hz |

N is unpadded (1--50); NN is zero-padded (01--50). EEG is read in its stored
microvolt scale for all days, with no additional Day 3 scale conversion. Channel
names are matched case-insensitively and the same 57 common channels are retained.
The public files are not replaced, re-extracted, or rewritten by this module.

EEG is resampled from 250 to 100 Hz. The stored 50-Hz wav2vec feature is resampled
to 100 Hz **in memory using `scipy.signal.resample_poly`**, the existing loader's
polyphase method; this is not linear interpolation or native 100-Hz extraction,
and it does not create new temporal information. Envelope remains at 100 Hz.
Arrays align at time zero and are cropped to their common minimum length across
the selected features. Switching the feature subset may change the crop length.

The original imported config declares word2vec unavailable. The later inspected
local release contains 50 static-word arrays, but that does not establish their
presence in the already published OpenNeuro 1.0.0 snapshot. The new external-config
helper can explicitly enable word2vec after checking every required input file.
Direct `--features word2vec` with the unchanged original config still raises an
unavailable-input error; BERT is not silently substituted.
`config_acoustic.yaml` is retained for compatibility, defaults to the same two
default inputs, and does not imply that acoustic word2vec has been supplied.

## Scientific settings retained from the source module

- Train separately for each participant and each recording day.
- Day 1: stories 1--17, 13 train / 2 validation / 2 test; Day 2: 18--33,
  12 / 2 / 2; Day 3: 34--50, 13 / 2 / 2.
- Subject/day-specific deterministic splits use seed 20260805; split files are
  saved with outputs. All features within a subject/day share its split.
- Common-average referencing; train-only normalization; standardized EEG
  clipping at +/-20. No new bad-segment rejection or interpolation is added.
- Ridge decoder, MSE validation, 0--500 ms EEG lag at 50-ms steps; seven alpha
  candidates from 0.0001 through 100. No PCA or feature-extraction step.
- Retrieval uses 3, 5, and 10-second nonoverlapping windows, one positive plus
  four same-trial negative candidates, and flattened Pearson correlation.
- Group summaries keep days separate; error bars show across-subject SEM.

## Run

Use Python 3.10 or later and an isolated environment. From `code/speech_retrieval`:

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B run_experiment.py --validate-only --output-dir ../../../pkueeg_retrieval_input_check
python -B run_experiment.py --subjects sub-01 --output-dir ../../../pkueeg_retrieval_new_run
python -B run_experiment.py --output-dir ../../../pkueeg_retrieval_full_run
python -B analyze_results.py --output-dir ../../../pkueeg_retrieval_full_run
```

The output-directory argument is relative to the current working directory.
By default, output is a sibling of the release named
`pkueeg_speech_retrieval_public_20260919`. Both experiment and plotting CLI reject
output directories inside the release (including resolved symlinks). Use a fresh
external directory per run and `-B` to prevent Python bytecode caches in the package.
Ensure that the output parent is writable. The input-check command reads EEG
headers/channel names and feature shapes/sampled values; it is not a complete
artifact-quality or scientific-validity check.

Outputs include `config_resolved.yaml`, `run_metadata.json`, per-subject/day
split/model/metric files, `summary_subject.csv`, and `summary_group.csv`. Plotting
adds statistics and feature-wise figures under the external output directory.

## Provenance and limitations

The historical `feature_inputs*.tsv` and `package_checksums.tsv` from the source
candidate do not describe these inputs or migrated code and are not distributed
as current manifests. Do not infer current data hashes from historical candidate
manifests or treat old scores as validation of OpenNeuro ds008834 version 1.0.0.
Use the public release's own metadata and inventories for its data provenance.
The current preprocessing limitations documented by the release remain unchanged.

`exp4/` retains the source package structure: data loading, ridge fitting,
retrieval evaluation, and dataset checks. `tests/` contains synthetic algorithm
and migration/portability tests. These are not formal experiment results.
