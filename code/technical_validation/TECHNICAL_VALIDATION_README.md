# PKUEEG technical validation code

For the standalone GitHub checkout, use the explicit external-input commands in
[RUNNING.md](../../docs/RUNNING.md). The package-relative examples and shell
launchers below describe code embedded in a dataset; they are not directly
usable with a separately downloaded dataset. The scientific Python code is
preserved. Keep outputs outside both the entire dataset and code checkout.

These scripts were copied from `code/technical_validation/` in the PKUEEG
release candidate dated 20260908 and adapted on 2026-09-19 to the public
`PKUEEG_release_20260909` layout. The numerical methods and historical defaults
are retained. This integration is **not a full rerun** and does not establish
that historical results reproduce on the public data. No historical results
or behavior data are included.

It reads the released 1-40 Hz derivative and 100 Hz speech envelopes. Default
paths are release-relative, without laboratory-specific absolute paths:

- Raw: `sub-XX/ses-dayX/eeg/*_task-audio_eeg.vhdr`.
- EEG: `derivatives/preproc_40hz/sub-XX/ses-dayX/eeg/` with filenames ending
  `_desc-storyNN_eeg.npz` (01-50, zero-padded) or `_desc-rest_eeg.npz`.
- Envelope: `derivatives/stimulus_features/envelope/envelope_100hz/N_envelope.npy`.

Released NPZ `eeg_data` values already use **microvolts for all days**. Do not
repeat the historical day-3 correction. MNE reads raw BrainVision into volts;
raw QC converts those values to microvolts. Channel matching is case-insensitive.
The scripts do not repair data or remove additional artifacts. QC scalp plots
are skipped with an explicit note because measured electrode coordinates are
not distributed. No coordinate data are fabricated.

The speech-segment retrieval/decoding experiment is provided separately in
`../speech_retrieval/`; see that directory's README for its environment and entry point.

## Included analyses

1. `pkueeg_completeness_temporal.py` checks 25 participants, 50 stories, 1,250
   participant-story recordings, 75 rest recordings, the 17/16/17 story
   allocation and EEG-envelope duration compatibility.
2. `pkueeg_validation_recheck.py` computes leave-one-out listening-versus-rest
   ISC and participant-level speech-envelope TRF waveforms, peak timing and
   cross-day waveform similarity.
3. `pkueeg_trf_prediction.py` performs nested, duration-balanced, story-disjoint
   cross-validation and calculates participant-level out-of-sample
   envelope-to-EEG correlations.
4. `pkueeg_behavior_analysis.py` optionally merges explicitly supplied external
   long-format accuracy values with the cross-validated neural
   scores, then reports Pearson correlations, Fisher 95% intervals,
   permutation p values and Benjamini-Hochberg q values. Behavior data are not
   distributed; this branch cannot run from the public release alone.
5. `pkueeg_low_neural_qc.py` reports low participant-day prediction values
   without automatically excluding them.
6. `data_quality/` contains the chunked raw-signal and preprocessed-signal
   quality workflow used to summarize amplitudes, flat/jump fractions, spectral
   measures, auxiliary-channel events and files requiring review.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv ../../../pkueeg_validation_env
source ../../../pkueeg_validation_env/bin/activate
python -m pip install -r pkueeg_requirements.txt
python -B -m unittest discover -p 'test_*.py'
python -B pkueeg_trf_prediction.py --self-test
```

## Run the analyses

From this directory:

```bash
bash run_pkueeg_technical_validation.sh
```

The default workflow runs completeness, prediction, ISC/TRF and low-neural-value
QC. It skips behavior because its data are not distributed. Raw signal-quality
analysis is substantially slower and is opt-in:

```bash
RUN_RAW_QUALITY=1 bash run_pkueeg_technical_validation.sh
```

Individual scripts accept command-line path overrides and can be run separately.
Generated tables and JSON files are written beside the release in
`<release-name>_outputs/technical_validation/`; figures are saved
as separate PNG and SVG files.

The imported TRF protocol uses -300 to 600 ms delays, 250 Hz analysis sampling,
and five outer folds. This is not the separate exploratory -200 ms protocol.
`preproc_8hz` is not the default input; alternate bands/sample rates require
matching explicit settings. A single-subject/day smoke command is:

```bash
python -B pkueeg_trf_prediction.py --subjects sub-01 --days day1 --n-jobs 1 \
  --output-dir ../../../pkueeg_validation_smoke
```

It still fits all 17 day-1 stories and is not a full replication.

## Optional external behavior input

No private or deferred-release directory is searched automatically. Supply an
authorized long-format TSV with `participant_id`, `session_id`, and
`comprehension_accuracy`: 75 rows for 25 participants x 3 sessions, IDs
`ses-day1` to `ses-day3`, accuracy fractions in [0, 1]. The wide table is not
accepted.

```bash
python -B pkueeg_behavior_analysis.py --behavior /path/to/authorized_behavior_long.tsv \
  --neural-summary /path/to/prediction/subject_day_summary.csv \
  --output-dir /path/to/new_behavior_results
```

In the combined script use `--analyses behavior --behavior <external TSV>` and
`--prediction-summary <new prediction summary>`. Requesting behavior without
its explicit input is an error, not permission to invent values.

## Statistical interpretation

ISC is calculated by correlating each participant's channel time series with the
mean of all other participants who heard the same story. Participant
correlations are averaged within channel, followed by Fisher-z averaging across
channels. Stories from the same participant and repeated uses of a daily rest
recording are not independent biological replicates. Prediction folds keep
entire stories disjoint. Cross-day comparisons are descriptive because day,
story set and acquisition system are confounded. Behavioral analysis uses only
explicitly supplied observed accuracy values and contains no simulated or target
correlations.

## Output safety update (2026-09-10)

Default analysis outputs now go to `<release-name>_outputs/technical_validation/`
beside the release, never inside the distributed code. CLI entry points reject
existing result directories/files. Choose a new output path for every new
analysis. Raw-quality paths are resolved relative to the configuration file;
`output_dir: null` selects the same external output policy. Completed raw-quality
subjects can be skipped only when their configuration fingerprint matches;
configuration changes or incomplete outputs require an explicit `--overwrite`
or a new directory. Aggregation also requires explicit overwrite permission.
Previous output-path examples above describe historical runs if they differ.
