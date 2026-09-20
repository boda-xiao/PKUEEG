# PKUEEG stimulus feature code

Start with the repository [feature-extraction guide](../../../docs/FEATURES.md)
for external data paths, dependency installation, CLI examples, model pins,
licensing and verification limits. This is a code distribution: audio,
annotations and time-series feature arrays must be obtained separately.

## Entry points and recipe documentation

| Operation | Entry point | Documentation |
|---|---|---|
| Envelope, Mel, Chinese wav2vec2 and optional static word features | `extract_features.py` | [Acoustic/static-word recipes](FEATURE_EXTRACTION_V2.md) |
| Mel only | `extract_mel.py` | [Acoustic/static-word recipes](FEATURE_EXTRACTION_V2.md) |
| Chinese wav2vec2 only | `extract_chinese_wav2vec.py` | [Acoustic/static-word recipes](FEATURE_EXTRACTION_V2.md) |
| Frozen fastText lexical subset, historically named word2vec | `extract_word2vec_100hz.py` | [Static word features](WORD2VEC_EXTRACTION.md) |
| BERT v3, all 12 word-level layers | `extract_bert_v3.py` | [BERT v3 recipe](BERT_V3_EXTRACTION.md) |
| BERT v3 layer-12 time rasterization | `rasterize_bert_v3.py` | [BERT v3 recipe](BERT_V3_EXTRACTION.md) |
| Existing acoustic/static-word array inspection | `verify_features.py` | [Verification scope](../../../docs/FEATURES.md#verification) |

`extract_envelope.py` supplies the envelope function to the shared driver; it
does not have its own command-line entry point. The shared driver defaults to
all four feature families, including wav2vec2, so explicitly select a subset
when no model directory is supplied. See the
[additive import record](FEATURE_EXTRACTION_IMPORT.md) for historical provenance.

Run commands from the repository root, use an explicit `--release-root` pointing
to the separately downloaded dataset, and select a new output directory outside
both the dataset and this repository. For example, after installing the acoustic
recipe dependencies in a separate Python 3.10 environment:

```bash
python -B code/feature_extraction/stimulus_features/extract_features.py --release-root /data/PKUEEG --features envelope mel --stories 1 --output-root /outputs/new_acoustic_story01
```

Replace the example absolute paths with your own. Model inference, word2vec,
BERT rasterization and optional reference comparisons have additional input
requirements; use the [full CLI guide](../../../docs/FEATURES.md#commands).

## What is and is not provided

The new acoustic recipes, word2vec extractor, BERT v3 extractor and rasterizer
are included. Their dependency files and model fingerprints are present, but
complete model-download workflows and large BERT/wav2vec2 weights are not.
The small `word2vec_lexicon.npz` is a pretrained-model subset distributed under
[CC BY-SA 3.0, not CC0](WORD2VEC_THIRD_PARTY_NOTICE.md).

No end-to-end MFA alignment pipeline is included. BERT v3 time rasterization
requires explicit external MFA v3 word intervals, excluded-year intervals and
an alignment manifest. Existing legacy timings are not silently substituted.
Technical-validation analysis code is included elsewhere in `code/`; analysis
results and behavioral-accuracy tables are not supplied by this code package.

The imported recipes do not establish numerical or bitwise equivalence to all
legacy released arrays. Local additions also do not change an already published
OpenNeuro snapshot. Check which inputs and references exist in your downloaded
dataset version before requesting verification.

## Retained legacy code and historical packaging

`extract_word_level_bert.py` remains the separate legacy BERT entry point. Its
input option is `--word-boundary-dir` (singular), unlike the v3 override
`--word-boundaries-dir`. The legacy script has an unpinned default remote model
and may download model files. It uses a different window/pooling implementation
and is not a verified reconstruction of all released MAT arrays. Earlier checks
reported a terminal overlapping-window difference for stories 1 and 50; this
documentation update does not rerun or change that algorithm.

`add_pkueeg_stimulus_derivatives.py` is archived under
`code/provenance/legacy_layout/`. It copies existing features and annotations and
converts historical layouts; do not run it against the current layout. Its
selection of every second legacy 100-Hz Mel/BERT frame for stories 34-43 was a
packaging operation, not extraction from audio or text.
