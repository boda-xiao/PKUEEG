# Publication preflight checks

Checked on 2026-09-20 for the initial companion-code upload. These checks concern
this code snapshot, not certification of all files in OpenNeuro version 1.0.0.
The original data release was not modified. Test outputs were kept in separate
working directories and are not included in this code-only repository.

## Automated tests

The following tests passed in an isolated copy of the code on Linux with
Python 3.10.20:

| Suite | Passed tests |
| --- | ---: |
| External retrieval-config helper | 13 |
| Technical-validation unit tests | 11 |
| Speech-retrieval unit tests | 16 |
| Stimulus-feature unit tests | 29 |
| **Total** | **69** |

Commands, run from the repository root unless a working directory is shown:

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s code/technical_validation -p 'test_*.py' -v
python -B code/technical_validation/pkueeg_trf_prediction.py --self-test

# Working directory: code/speech_retrieval
python -B -m unittest discover -s tests -v

# Working directory: code/feature_extraction/stimulus_features
python -B -m unittest discover -p 'test_*.py' -v
```

The additional TRF synthetic recovery test passed, with prediction correlations
of 0.964890, 0.954925, 0.975008 and 0.969807 for its four simulated channels.
These are synthetic-test results, not PKUEEG participant results.

## Real-data input check

A new external retrieval configuration enabled envelope, wav2vec and word2vec.
Running `run_experiment.py --subjects sub-01 --validate-only` against the current
local data package passed. The validator checked EEG headers/channel consistency
for this participant and loaded 50 stimulus arrays for each selected feature:

| Feature | Dimensions | Stored sampling rate |
| --- | ---: | ---: |
| Envelope | 1 | 100 Hz |
| wav2vec | 1024 | 50 Hz |
| Static word embeddings, named word2vec in the code | 300 | 100 Hz |

Retrieval resamples the stored wav2vec arrays to its 100-Hz analysis grid in
memory. This check did not fit a decoder, validate all participants' samples, or
establish that all optional local files are present in the public OpenNeuro
snapshot.

## Tested environment

The existing scientific environment contained NumPy 2.2.6, SciPy 1.15.3,
PyYAML 6.0.3, pandas 2.3.3, matplotlib 3.10.9, MNE 1.12.1,
scikit-learn 1.7.2, threadpoolctl 3.6.0, soundfile 0.14.0, librosa 0.11.0,
PyTorch 2.6.0+cu124 and transformers 5.12.0. This is a record of the test
environment, not a dependency lock or an assertion that every installed package
was exercised. Seaborn was not installed in that environment; the supplied
analysis requirements include it, but ISC plotting was not tested here.

## Upload integrity and limits

The [source manifest](SOURCE_MANIFEST.json) records SHA-256 checksums for every
imported file. Scientific Python/shell code, original configurations, dependency
lists and the licensed lexical asset retain their source bytes. Eight imported
Markdown files were updated to explain standalone use and historical limitations.
Root-level documentation, the external-config helper and its tests are additions.

The upload audit checks Python syntax, relative Markdown links, common credential
patterns, file sizes and the absence of EEG arrays, time-series features and
model weights. The only included binary asset is the attributed static-word
lexical subset described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
A pattern scan is not a guarantee that every possible secret can be detected.

Not performed as part of this upload: a clean dependency installation, full
25-participant experiments, behavior-neural analysis, raw-data preprocessing,
GPU feature extraction, a fresh MFA alignment, or a byte-exact comparison of
regenerated derivatives. See [FEATURES.md](FEATURES.md) and
[RUNNING.md](RUNNING.md) for prerequisites and known boundaries.
