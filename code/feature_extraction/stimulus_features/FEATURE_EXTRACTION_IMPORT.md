# Additive feature-extraction code import

Imported on 2026-09-19 from the `PKUEEG_release_candidate_20260908` code recipes.
All new files are placed directly in the existing
`code/feature_extraction/stimulus_features/` directory. No existing file was
replaced, no directory was added or moved, and no released feature or EEG array
was regenerated in place. Existing README files, retrieval settings, metadata,
model-download placeholders, and the legacy BERT/word2vec scripts remain intact.
Their older statements about missing extraction code may therefore be stale;
this document describes only the additive import. OpenNeuro 1.0.0 is unchanged.

## Entry points

| Feature / operation | Entry point | Notes |
|---|---|---|
| Envelope | `extract_envelope.py`, called by `extract_features.py` | Original 0908 compressed gammatone recipe; 100 Hz. |
| Mel | `extract_mel.py`, called by `extract_features.py` | Original 0908 log-power Mel recipe; 100 and 50 Hz outputs. |
| Chinese wav2vec2 | `extract_chinese_wav2vec.py`, called by `extract_features.py` | Frozen Chinese model, layer 9; approximately 50 Hz native output plus 100 Hz interpolation. |
| word2vec | Existing `extract_word2vec_100hz.py`, also callable by `extract_features.py` | Existing frozen fastText lexical subset and 100 Hz recipe; not modified in this import. |
| New word-level BERT | `extract_bert_v3.py` | The 0908 v3 fixed WordPiece-window recipe, all 12 layers. |
| New time-aligned BERT | `rasterize_bert_v3.py` | Layer 12 on 50/100 Hz grids; requires explicit external MFA v3 alignment outputs. |
| Feature verification | `verify_features.py` | See its help and the v2 documentation for the precise integrity/comparison scope. |

`extract_word_level_bert.py` is the retained legacy script. It is not renamed,
replaced, or silently redirected to the new v3 implementation.

## Read the recipe-specific instructions

- `FEATURE_EXTRACTION_V2.md`: envelope, Mel, Chinese wav2vec2 and static-word
  extraction, dependencies, model pins, output safety, and reference comparison.
- `BERT_V3_EXTRACTION.md`: word-level BERT, offline model requirements, and
  time rasterization with explicit alignment input.
- `WORD2VEC_EXTRACTION.md`: the earlier word2vec import and exact reproduction
  test, which is not extended to other features by this code-only addition.
- `FEATURE_CODE_IMPORT.json`: the imported file inventory and SHA-256 hashes.

The source computational recipes are preserved. Adaptations concern portable
0909 paths, uniquely named flat-directory modules, offline input selection,
output protection, provenance, and verification/reporting. Check individual
recipe documentation before running. New outputs must be written to a fresh
directory outside the release; the scripts do not promote generated files into
the package or modify downstream experiment defaults.

## Dependencies and limitations

The import provides code, dependency lists, small model-pin metadata and
applicable notices; it does not bundle new model weights, install packages,
download models, or rerun all downstream experiments. Chinese wav2vec2 and BERT
need external offline pretrained snapshots matching their recorded hashes.
The gammatone envelope additionally needs Brian2Hears, Cython and a C/C++ compiler.
Use the supplied requirements in a separate environment rather than changing
an environment used by another experiment.

The 0909 word tables can supply the lexical sequence for BERT extraction.
The v3 time-aligned BERT step additionally requires MFA v3's
`alignment_manifest.json`, word interval tables and excluded-year tables.
Those inputs and the MFA workflow are not added by this feature-code import.
Legacy 0909 timings are not silently treated as equivalent to independent MFA
v3 alignment. Numeric years must remain outside the lexical sequence.

The 0908 refreshed recipes and the stored 0909 legacy arrays are not generally
interchangeable. In particular, model choice, windowing, frame count, timing,
and the native-versus-interpolated sampling grid can differ. Including code
does not establish that it exactly reconstructs every stored 0909 array. A
reference mismatch is reported, not fixed by overwriting/truncating/resampling
the stored data. No all-feature or downstream-result equivalence is claimed.

Third-party model licenses remain applicable; see the new recipe-specific
notices and the pre-existing `WORD2VEC_THIRD_PARTY_NOTICE.md`. The package-level
license has not been edited or taken as permission to relicense model assets.
