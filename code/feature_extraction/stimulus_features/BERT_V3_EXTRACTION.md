# BERT v3 feature extraction imported from the 0908 release

This additive module preserves the scientific recipe from
`PKUEEG_release_candidate_20260908/code/mfa_bert_v3/extract_bert.py` and
`rasterize_bert.py`. Existing 0909 scripts, features, annotations and README files
are not replaced. This is a feature-extraction import, not an MFA alignment
pipeline and not a replacement for `extract_word_level_bert.py`.

## Files and inputs

- `extract_bert_v3.py`: word-level extraction, using all 12 hidden layers of the
  frozen `google-bert/bert-base-chinese` checkpoint. Its default lexical input is
  the existing 0909 `derivatives/stimulus_annotations/word_boundaries/legacy/`
  directory. Use `--word-boundaries-dir` to select another explicit TSV source.
- `rasterize_bert_v3.py`: layer-12 embeddings rasterized at 50 and 100 Hz using
  explicit MFA v3 word/year intervals. It requires an external `--alignment`
  directory containing `alignment_manifest.json`, `words/story-NN_words.tsv`
  and `years/story-NN_years.tsv`. These MFA v3 inputs are not bundled in 0909.
  Legacy timings are not silently substituted and missing year intervals are
  never fabricated.
- `bert_v3_common.py`: shared input/output guards and fingerprints.
- `bert_v3_model_checksums.json`: pinned model revision and SHA-256 fingerprints.
- `requirements_bert_v3.txt`: dependency versions from the tested 0908 recipe.
- `test_bert_v3.py`: bounded tests without model downloads or GPU execution.
- `BERT_V3_APACHE-2.0.txt` and `BERT_V3_THIRD_PARTY_NOTICES.md`: model notices.

Model weights are external and must be provided through `--model-dir`; there is
no automatic network download. The pinned revision is
`8f23c25b06e129b6c986331a13d8d025a92cf0ea`. Every listed model file is checked before
loading. The exact same checkpoint and software versions are necessary but do
not by themselves guarantee cross-device bitwise equivalence.

## Commands

Run from the release root. The example paths are placeholders outside the
release; replace them with your own existing model/alignment directories and
new output locations. Outputs must be disjoint from the release and all inputs,
and the output directory must not already exist.

```bash
python -B code/feature_extraction/stimulus_features/extract_bert_v3.py \
  --model-dir ../model_cache/bert \
  --output-dir ../new_bert_words --device cuda:0

python -B code/feature_extraction/stimulus_features/rasterize_bert_v3.py \
  --word-vectors ../new_bert_words \
  --alignment ../external_mfa_v3_alignment \
  --output-dir ../new_bert_time_aligned

python -B code/feature_extraction/stimulus_features/test_bert_v3.py
```

Use `--stories 1` for a bounded word-level extraction smoke test. The rasterizer
processes the stories listed in the resulting word-feature manifest and requires
matching ordered lexical words in the explicit alignment input.

## Scientific recipe and limits

Words are concatenated without punctuation or excluded numeric years. BERT is
bidirectional and noncausal. WordPiece windows contain at most 510 tokens with a
384-token stride. Overlapping contextual occurrences are averaged per token;
tokens are then averaged per lexical word. All twelve hidden layers are saved
as a deterministic-header MAT file with shape `(12, number_of_words, 768)`.

The time-aligned recipe uses layer 12, half-open intervals
`onset <= sample_time < offset`, zeros outside words and during explicit excluded
year intervals, and `ceil(duration_seconds * sampling_rate)` frames. It does not
interpolate BERT embeddings. Overlapping retained words or year/word intervals
fail validation.

Portability changes are limited to prefixed filenames/imports, the existing 0909
lexical path, an explicit external model directory, selectable story IDs and
additional lexical-order fingerprints. The extraction/window/pooling and
rasterization algorithms are unchanged. No existing 0909 BERT feature is
replaced. This v3 recipe does **not** claim bitwise or numerical equivalence to
the pre-existing 0909 legacy word-level or time-aligned BERT features.

