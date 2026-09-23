# PKUEEG: processing and analysis code

Code accompanying **PKUEEG**, a continuous Mandarin speech-perception EEG
dataset with 25 participants, three recording days, and 50 stories (17/16/17
stories by day).

- **Data:** [PKUEEG on OpenNeuro](https://openneuro.org/datasets/ds008834)
- **Published data snapshot:** [ds008834, version 1.0.3](https://openneuro.org/datasets/ds008834/versions/1.0.3)
- **Code:** [boda-xiao/PKUEEG](https://github.com/boda-xiao/PKUEEG)
- **Instructions:** [running the analyses](docs/RUNNING.md),
  [feature extraction](docs/FEATURES.md), and
  [OpenNeuro cross-link text](docs/OPENNEURO_LINKING.md).

## Scope and version compatibility

This is a **code-only companion repository**, prepared from the locally maintained
`PKUEEG_release_20260909` code as inspected on 2026-09-20. It includes later
2026-09-19 code imports from an earlier candidate. The local package name is not
an OpenNeuro version identifier: these additions do not retroactively change
the already published OpenNeuro 1.0.0 snapshot.

The repository does not contain raw EEG, preprocessed EEG, audio, transcripts,
time-series stimulus features, behavioral accuracy, pretrained neural-model
weights, or historical experiment results. Download the dataset separately.
A small frozen fastText lexical subset is included as an explicitly licensed
exception; it is not an EEG/feature-time-series dataset.

| Component | Location | Important qualification |
| --- | --- | --- |
| EEG preprocessing | `code/preprocessing/` | Legacy entry point with documented limitations; not a byte-exact reproduction guarantee. |
| Envelope, Mel, Chinese wav2vec2, word2vec and BERT extraction | `code/feature_extraction/stimulus_features/` | New recipes may differ from stored legacy arrays; neural models are external. |
| ISC, envelope TRF, EEG prediction and signal QC | `code/technical_validation/` | Explicit external data paths are needed for this code-only checkout. |
| Behavior-neural association | `code/technical_validation/` | Requires an authorized external behavioral table, not supplied here. |
| Five-candidate speech retrieval | `code/speech_retrieval/` | Default features are envelope and wav2vec; word2vec is an explicit opt-in. |
| BIDS/layout checks | `code/validate_layout.py`, `code/run_bids_validation.py` | Structural validation is not scientific validation. |
| Historical packaging code | `code/provenance/legacy_layout/` | Provenance only; not entry points for the current layout. |

**MFA is not bundled as an executable alignment workflow in this checkout.**
Time-aligned BERT v3 consumes separately obtained, validated MFA-v3 alignment
outputs. Existing legacy annotations cannot silently replace those inputs.

The inspected local package contains 50 later-added `word2vec_100hz` arrays;
their presence does not establish that OpenNeuro 1.0.0 contains them. The original
two-feature retrieval configuration is preserved. The configuration helper below
enables word2vec only when explicitly requested and all required files exist.

## Quick start

Examples use a Linux shell and Python 3.10. Keep three separate directories:
the code checkout, the downloaded dataset, and new analysis outputs. None should
contain another. Do not write results or generated features into the dataset.

```bash
git clone https://github.com/boda-xiao/PKUEEG.git
cd PKUEEG
python -m venv ../pkueeg-analysis-env
source ../pkueeg-analysis-env/bin/activate
python -m pip install -r requirements/analysis.txt

# Edit these two paths for your machine.
export PKUEEG_DATA="/path/to/downloaded/PKUEEG"
export PKUEEG_OUT="/path/to/new/PKUEEG-results"
mkdir -p "$PKUEEG_OUT"

# Make a new external config without changing the distributed defaults.
python -B scripts/prepare_retrieval_config.py \
  --dataset-root "$PKUEEG_DATA" \
  --output-config "$PKUEEG_OUT/retrieval.yaml" \
  --output-dir "$PKUEEG_OUT/retrieval-run"

# Input checks only: this does not train a decoder.
python -B code/speech_retrieval/run_experiment.py \
  --config "$PKUEEG_OUT/retrieval.yaml" --validate-only \
  --output-dir "$PKUEEG_OUT/retrieval-input-check"
```

Use a fresh output path for each command. Input checking itself creates a report
directory. Full training, plotting, TRF, preprocessing, and feature-extraction
commands are documented in [RUNNING.md](docs/RUNNING.md) and
[FEATURES.md](docs/FEATURES.md). Read the task-specific dependencies before
running a different task: the legacy `code/requirements.txt` is not a complete
environment, and neural feature extraction needs a separate environment.

The preserved shell launchers assume code embedded inside the dataset. For a
separate GitHub checkout, use the explicit-path commands in these documents.

## Tests and reproducibility boundaries

Run the data-free unit tests and the synthetic TRF recovery check:

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s code/technical_validation -p 'test_*.py' -v
python -B code/technical_validation/pkueeg_trf_prediction.py --self-test
(cd code/speech_retrieval && python -B -m unittest discover -s tests -v)
```

Feature-recipe tests require that recipe's environment. Test results are summarized
in [publication checks](docs/PUBLICATION_CHECKS.md). Synthetic tests, model-file
hash checks and successful imports do **not** establish reproduction of all
released arrays or manuscript results. Full scientific experiments were not
rerun as part of the GitHub upload. Preserve the exact dataset snapshot, code
commit, runtime environment, input hashes, split files, and resolved configuration
when reporting new analyses.

Known limitations include legacy preprocessing channel-name matching, no explicit
bad-segment rejection or saved per-session ICA records in that entry point,
differences between legacy and refreshed feature recipes, incomplete model-download
automation, and external MFA/behavioral inputs. Some legacy output guards protect
the checkout and specified input directories, not every sibling directory in an
external dataset. Always choose an output location outside the entire dataset
and checkout; do not override it to an input location.

## Licensing

The existing release's CC0 declaration is retained for PKUEEG-authored material;
see [LICENSE](LICENSE). **It does not override third-party licenses.** In particular,
`word2vec_lexicon.npz` contains fastText-derived vectors under **CC BY-SA 3.0**.
Model-specific terms and attribution are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Neural model weights are not
redistributed here. This repository does not grant new rights to separately
downloaded stimulus recordings or texts.

## Citation and support

For the data, copy the citation and DOI from the exact
[OpenNeuro snapshot](https://openneuro.org/datasets/ds008834/versions/1.0.0)
you actually used. For code, cite this repository together with its commit hash
or a subsequently published code release. The code and dataset have independent
version histories. No code DOI or paper DOI is asserted here.

Report software issues through [GitHub Issues](https://github.com/boda-xiao/PKUEEG/issues),
including the code commit, dataset snapshot, command, environment, and a minimal
error message. Do not attach credentials, participant identifiers, or restricted
behavioral records.
