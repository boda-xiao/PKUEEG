# Stimulus feature extraction

This repository contains feature-extraction code, not the EEG, audio,
annotations or time-series feature arrays. Obtain a compatible dataset
separately and pass its root explicitly. The additive code imports do not update
an already published OpenNeuro snapshot and do not establish that new outputs
reproduce every historical array. No full feature-generation or scientific
rerun is claimed by this documentation update.

## Recipes and inputs

| Feature | Input | Output recipe |
|---|---|---|
| Envelope | `stimuli/audio/story-NN.mp3` | 28 ERB-spaced gammatone bands, 50-5000 Hz; compressed-band sum; float64 at 100 Hz |
| Mel | Same audio | 80 Slaney log10-power bands; 16-kHz audio, FFT 512, Hann 400, hop 160; float64 at 100 Hz and alternate frames at 50 Hz |
| Chinese wav2vec2 | Same audio plus external pinned checkpoint | Hidden state 9, 1024 dimensions; float32, approximately 50-Hz native chunk output and FFT-interpolated 100-Hz output |
| Static word features, named `word2vec` | Audio duration, `derivatives/stimulus_annotations/word_boundaries/legacy_mat/story-NN_word-times.mat`, frozen lexical subset | 300-dimensional float32 at 100 Hz; nearest-centisecond half-open word intervals, zero elsewhere |
| BERT v3 word-level | `derivatives/stimulus_annotations/word_boundaries/legacy/story-NN_words.tsv` plus external pinned checkpoint | All 12 layers; float32 MAT array `(12, n_words, 768)` |
| BERT v3 time-aligned | BERT v3 word outputs plus external MFA v3 alignment | Layer 12, 768 dimensions at 50 and 100 Hz; no interpolation; zero outside retained words and during excluded-year intervals |

The name `word2vec` is historical: the actual static vectors derive from
fastText Chinese Common Crawl/Wikipedia `cc.zh.300.bin`. Rasterization uses the
bundled frozen vocabulary, not a new model training run. The full fastText model
and fastText Python library are not needed for this extractor.

BERT v3 concatenates lexical words without punctuation, uses at most 510
WordPieces per window with stride 384, averages repeated contextual occurrences
per token, then averages tokens per word. It is bidirectional and noncausal.
Lexical words must be nonempty and contain no digit characters; numeric years
must not be inserted into this word sequence. The legacy BERT script is retained
as a different, unpinned recipe, not an alias for v3.

Full recipe definitions and historical import records:

- [Acoustic/static-word recipes](../code/feature_extraction/stimulus_features/FEATURE_EXTRACTION_V2.md)
- [Static word extraction](../code/feature_extraction/stimulus_features/WORD2VEC_EXTRACTION.md)
- [BERT v3 extraction and rasterization](../code/feature_extraction/stimulus_features/BERT_V3_EXTRACTION.md)
- [Additive feature-code import](../code/feature_extraction/stimulus_features/FEATURE_EXTRACTION_IMPORT.md)

## Environment and models

Use separate Python 3.10 environments for the recipe being run. The root
`code/requirements.txt` is not a complete extraction environment. The supplied
recipe requirements pin direct dependency versions, but are not a full lock of
transitive dependencies, operating system, compiler, GPU driver or CUDA runtime.
Brian2Hears envelope extraction requires Cython and a working C/C++ compiler.
Use a compatible PyTorch build for your CPU/GPU; retaining pinned versions alone
does not guarantee cross-device bitwise equality.

The new BERT and wav2vec2 extractors use `local_files_only=True` and do not
download models. `code/model_downloads/` is currently a status directory, not an
implemented downloader. Acquire the required files from their official upstream
repositories, retain their licensing information, and provide dedicated local
model directories matching the recorded hashes.

### Chinese wav2vec2

Upstream: [TencentGameMate/chinese-wav2vec2-large](https://huggingface.co/TencentGameMate/chinese-wav2vec2-large).
Its model card declares MIT. The script pins these file hashes, but does not
record an upstream commit revision:

| File | SHA-256 |
|---|---|
| `pytorch_model.bin` | `c8a5554a79c3bbbe76f2e43d3d4b4369c8c2abd5515e623192e0381d7e5e7b3f` |
| `config.json` | `53ba47fee1b3630c489e2525af7102c74f05d23dbdd49b9265ff809444c0eabb` |
| `preprocessor_config.json` | `d325e3677f9bdbd1086f9f1eccae922b82c971b8a250085248452df1ac621701` |

Use a dedicated directory containing only these model files, plus optional
license/README text. Do not mix in `model.safetensors`, sharded weight indexes or
other alternate weights. The current extractor verifies `pytorch_model.bin` but
does not explicitly force that format when loading. Transformers can prefer
safetensors when present, so an extra unverified weight file could otherwise be
loaded instead of the checked binary. See the
[official loading documentation](https://huggingface.co/docs/transformers/main/models).

### BERT v3

Upstream: [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese),
declaring Apache-2.0. The pinned revision is
`8f23c25b06e129b6c986331a13d8d025a92cf0ea`. Required files are
`model.safetensors`, `config.json`, `vocab.txt`, `tokenizer.json` and
`tokenizer_config.json`; their SHA-256 values are in
[bert_v3_model_checksums.json](../code/feature_extraction/stimulus_features/bert_v3_model_checksums.json).
The default manifest is checked before loading. Keep that manifest unchanged
when following the documented pinned recipe. See the
[BERT third-party notice](../code/feature_extraction/stimulus_features/BERT_V3_THIRD_PARTY_NOTICES.md)
and accompanying Apache-2.0 text.

### Static word subset and licensing exception

`word2vec_lexicon.npz` is a 6,144-word, 300-dimensional pretrained-model subset,
not a time-series feature array. Its enforced SHA-256 is
`87aa3ff9460f9e1ad2a2c20efc6ecd210149f26dbb31820fac49843cb69ba76c`.
It retains CC BY-SA 3.0 terms, including attribution and ShareAlike, and is not
relicensed under any dataset CC0 declaration or repository code license.
Preserve its [third-party notice](../code/feature_extraction/stimulus_features/WORD2VEC_THIRD_PARTY_NOTICE.md)
and [model metadata](../code/feature_extraction/stimulus_features/word2vec_lexicon.json).
The notice describes vocabulary selection and the recovered character-average
fallback. See the [official fastText model license](https://fasttext.cc/docs/en/crawl-vectors.html)
and [CC BY-SA 3.0 terms](https://creativecommons.org/licenses/by-sa/3.0/).
The fastText software license is not the pretrained vectors' license.

## Commands

Run these examples from the repository root. Replace `/data/PKUEEG`, `/models`,
`/alignments` and `/outputs` with your own absolute paths. Output directories must
be new and should be outside both the downloaded dataset and this repository;
also keep them disjoint from model and alignment inputs. Existing outputs are
not overwritten. `--stories 1` is a bounded example; omit it to request all 50
stories. The examples are commands to run, not a report of tests performed here.

Install acoustic dependencies in a separate environment and generate envelope
and Mel without downloading or loading a pretrained model:

```bash
python -m pip install -r code/feature_extraction/stimulus_features/feature_extraction_v2_requirements.txt
python -B code/feature_extraction/stimulus_features/extract_features.py --release-root /data/PKUEEG --features envelope mel --stories 1 --output-root /outputs/new_acoustic_story01
```

For Mel alone, the standalone entry point is:

```bash
python -B code/feature_extraction/stimulus_features/extract_mel.py --release-root /data/PKUEEG --stories 1 --output-root /outputs/new_mel_story01
```

For Chinese wav2vec2, supply the prepared model directory. Change `cpu` to an
available device such as `cuda:0` if appropriate:

```bash
python -B code/feature_extraction/stimulus_features/extract_chinese_wav2vec.py --release-root /data/PKUEEG --model-dir /models/chinese-wav2vec2-pinned --device cpu --stories 1 --output-root /outputs/new_wav2vec_story01
```

The shared driver defaults to all four families, including wav2vec2 and
word2vec. Use `--features` explicitly for a subset; `--model-dir` is required
whenever wav2vec is selected. word2vec is optional for the acoustic-only commands
above and has its own smaller dependency list:

```bash
python -m pip install -r code/feature_extraction/stimulus_features/requirements_word2vec.txt
python -B code/feature_extraction/stimulus_features/extract_word2vec_100hz.py --release-root /data/PKUEEG --stories 1 --output-dir /outputs/new_word2vec_story01
```

Word-level BERT v3 requires lexical TSVs and the pinned model, but does not run
MFA and does not require the external MFA v3 interval tables:

```bash
python -m pip install -r code/feature_extraction/stimulus_features/requirements_bert_v3.txt
python -B code/feature_extraction/stimulus_features/extract_bert_v3.py --release-root /data/PKUEEG --model-dir /models/bert-base-chinese-pinned --device cpu --stories 1 --output-dir /outputs/new_bert_words_story01
```

Use `--word-boundaries-dir /path/to/lexical_tsvs` to override the BERT v3 lexical
input directory. The retained legacy script instead uses the singular
`--word-boundary-dir`; it is not covered by the v3 model-pin guarantee and can
download its default unpinned remote model.

### BERT time rasterization requires external MFA v3 inputs

There is no end-to-end MFA alignment workflow, MFA environment, or MFA model
bundle in this repository. No command here creates the missing alignment.
Only run the rasterizer after obtaining and independently validating:

- `alignment_manifest.json`, with `stories` entries containing `story` and `duration_seconds`;
- `words/story-NN_words.tsv`, with ordered `word`, `onset` and `offset` fields;
- `years/story-NN_years.tsv`, with the excluded-year `onset` and `offset` intervals.

Word order must match the lexical sequence fingerprint saved by the BERT v3
extractor. Year tables are required, even when they contain no intervals;
missing tables are not fabricated. Legacy word boundaries are not silently
treated as independent MFA v3 alignments.

```bash
python -B code/feature_extraction/stimulus_features/rasterize_bert_v3.py --release-root /data/PKUEEG --word-vectors /outputs/new_bert_words_story01 --alignment /alignments/mfa_v3 --output-dir /outputs/new_bert_time_story01
```

The rasterizer processes the stories in the word-feature manifest. Its grids
have `ceil(duration_seconds * rate)` frames and use half-open intervals. It
rejects word overlap and year/word overlap on the sampled grid. This is not a
complete external-alignment validator: validate finite, ordered, nonnegative,
in-duration intervals and audio durations separately before running it.

### Output layout

The shared driver writes `features/`, per-feature `extraction_records/` and an
`extraction_manifest.json`. Envelope files are named `{N}_envelope.npy`;
other families use `story-NN_...npy`. Standalone Mel writes timestamps under
`timestamps/`, whereas the shared driver uses `features/mel_timestamps/`.
Standalone wav2vec2 writes both rate directories under `features/`.

Standalone word2vec writes `story-NN_word2vec.npy` directly in its output
directory, plus `extraction_verification.json`. BERT word extraction writes
`story-NN_bert-layers01to12.mat` and `bert_word_manifest.json`; rasterization
writes `bert_50hz/`, `bert_100hz/`, corresponding `year_masks_.../` directories
and `bert_time_manifest.json`. These outputs are not automatically installed
into a dataset or enabled in downstream retrieval configurations.

## Verification

Generation does not need reference feature arrays. It reads audio and/or
annotations, plus the requested pretrained model or lexical subset. Reference
arrays are read only when optional comparison is requested after generation.

- Add `--verify` to standalone word2vec only if the selected dataset actually
  contains the corresponding `derivatives/stimulus_features/word2vec_100hz/`
  arrays. It requires identical array bits for success and separately reports
  NPY file-byte equality; a missing reference raises an error.
- The shared driver's `--verify` compares hashes of available same-rate
  references without interpolation, clipping or truncation. Outputs without
  counterparts are marked `NOT_COMPARABLE`. A differing comparable file, or
  no comparable files, produces a nonzero exit after saving its manifest.
- `verify_features.py` checks existing envelope, Mel, wav2vec2 and word2vec
  arrays for shape and finite values and records current hashes. It expects
  all four families for the requested stories; older snapshots without
  word2vec will fail this check. It is not a historical reproduction test.

For a dataset containing all four reference families:

```bash
python -B code/feature_extraction/stimulus_features/verify_features.py --release-root /data/PKUEEG --stories 1 --report /outputs/new_feature_structure_story01.json
```

Bounded unit tests can be run after installing the needed recipe dependencies:

```bash
python -B -m unittest discover -s code/feature_extraction/stimulus_features -p "test_*.py" -v
```

These tests cover selected recipe, geometry, serialization and output-safety
behaviors; they do not execute full pretrained-model inference, all feature
generation or downstream scientific analyses.

## Historical-equivalence limits

The imported recipes and legacy stored arrays are not generally interchangeable.
Mel can have a different terminal frame count. Wav2vec2 checkpoint choice,
three-second chunking and approximate native frame geometry can differ from
legacy extraction; its 100-Hz output is interpolated, not native 10-ms model
output. BERT v3 uses WordPiece windows and per-token then per-word pooling;
legacy BERT uses a different character-window/pooling implementation.
MFA v3 interval semantics are also distinct from legacy timing inputs.

A reference mismatch is a recipe/version difference to investigate, not a
reason to overwrite or silently reshape released arrays. Successful structural
checks, bounded tests or matching files for one family do not certify complete
scientific reproducibility or equivalence of downstream results.
