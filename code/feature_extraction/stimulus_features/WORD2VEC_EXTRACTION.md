# Imported 100 Hz static word features

This additive import was made on 2026-09-19 from
`PKUEEG_release_candidate_20260908`. It adds the extraction code and 50 existing
feature arrays to existing directories in `PKUEEG_release_20260909`.
No pre-existing file, README, configuration, annotation, EEG array, or other
feature array was edited. No directory was added or moved. Older availability
statements and the existing retrieval configuration have deliberately been left
unchanged; in particular, the default retrieval configuration does not enable
word2vec automatically. This import does not update OpenNeuro version 1.0.0.

## Feature definition and provenance

The historical name `word2vec_100hz` is retained. The actual model is the
**fastText Chinese Common Crawl/Wikipedia `cc.zh.300.bin`** static lexical model,
not a newly trained word2vec model. Each file contains a time-by-300 float32
array at 100 Hz. Word onset and offset times are rounded to the nearest
centisecond with `numpy.rint`; each half-open word interval is filled with its
static vector, and other frames are zero. The number of frames is the ceiling
of the decoded MP3 duration times 100. The extraction algorithm is unchanged
from the 0908 script; only package-relative input/default paths were adapted.

Inputs already present in this release:

- `stimuli/audio/story-NN.mp3`
- `derivatives/stimulus_annotations/word_boundaries/legacy_mat/story-NN_word-times.mat`

All 50 audio files and all 50 word-time MAT files were verified byte-identical
between the two packages. This is reproduction from the provided word timings,
not a rerun of forced alignment. No EEG or reference feature array is read
during feature generation. Reference arrays are read only when `--verify` is
requested, after each new array has been generated and saved.

Files added alongside this document:

- `extract_word2vec_100hz.py`: path-adapted extraction entry point.
- `test_word2vec.py`: unchanged source unit tests.
- `requirements_word2vec.txt`: minimal pinned extraction dependencies.
- `word2vec_lexicon.npz`: frozen, pickle-free 6,144-word, 300-dimensional model subset.
- `word2vec_lexicon.json`: unchanged source model metadata.
- `WORD2VEC_THIRD_PARTY_NOTICE.md`: unchanged source attribution and license notice.

The lexicon was recovered from the archived 0908 asset and checked against the
SHA-256 pinned by the source script:
`87aa3ff9460f9e1ad2a2c20efc6ecd210149f26dbb31820fac49843cb69ba76c`.
Its original notice calls it `assets/word_lexicon.npz`; in this flat, existing
0909 directory the same bytes are named `word2vec_lexicon.npz`. The lexicon is
a pretrained-model subset, not reconstructed from the time-series features.
The full fastText training corpus/model and forced-alignment environment are
not required to rasterize this frozen vocabulary and are not bundled here.

The 50 copied arrays and a machine-readable import manifest are located at
`derivatives/stimulus_features/word2vec_100hz/`. The manifest records original
sources, array hashes, and both packages' input hashes.

## Reproduce outside the release

From the release root, using Python 3.10:

```bash
python -m pip install -r code/feature_extraction/stimulus_features/requirements_word2vec.txt
python -B code/feature_extraction/stimulus_features/test_word2vec.py
python -B code/feature_extraction/stimulus_features/extract_word2vec_100hz.py \
  --output-dir ../word2vec_reproduced_new --verify
```

The output directory must not exist and must be outside the release. The
default processes all 50 stories; `--stories 1 2` selects a subset. The script
writes new arrays and `extraction_verification.json` only to the requested
external output directory. The report distinguishes numeric equality, exact
array-bit equality, and complete NPY file-byte equality. Verification exits
with an error if any generated array is not bit-identical.

## Third-party terms

The source notice identifies the upstream vectors and lexical subset as
**CC BY-SA 3.0, not CC0**. Read `WORD2VEC_THIRD_PARTY_NOTICE.md` before public
distribution. The release-level CC0 statement has not been changed by this
additive operation and must not be interpreted as relicensing this model asset.
This local addition is not an authorization or compatibility assessment for
uploading the assets to OpenNeuro.
