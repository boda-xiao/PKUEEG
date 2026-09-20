# Additive 0908 acoustic feature extraction code for the 0909 layout

These files are adapted from `code/stimulus_extraction_v2` in the 0908 release
candidate. Only release-relative input paths, external output protection,
reference checking and accompanying documentation were adapted. The numerical
envelope/Mel/Chinese wav2vec recipes are unchanged. Existing 0909 code, data,
configurations and documentation are not rewritten by this import.

## Scope and limitations

| Feature | Imported recipe | Current 0909 reference |
|---|---|---|
| Envelope | 28 ERB-spaced Brian2Hears gammatone bands, 50--5000 Hz; sum(abs(band)**0.6); float64; polyphase to 100 Hz | `derivatives/stimulus_features/envelope/envelope_100hz` |
| Mel | 80 Slaney log10-power bands; 16-kHz mono waveform; FFT 512, Hann 400, hop 160; float64; 100 Hz and every second frame for 50 Hz | Only legacy `derivatives/stimulus_features/mel/mel_50hz` is available for comparison |
| Chinese wav2vec | Frozen `TencentGameMate/chinese-wav2vec2-large`, hidden_states[9], 1024 dimensions; native approximately 50 Hz plus MNE FFT interpolation to 100 Hz | Only legacy `derivatives/stimulus_features/wav2vec2/wav2vec2_layer9_50hz` is available for comparison |
| Static word features | Reuses the pre-existing 300-dimensional frozen fastText lexical subset and installed word2vec extractor | `derivatives/stimulus_features/word2vec_100hz` |

**Adding these scripts does not prove that they reproduce the legacy 0909
envelope, Mel or wav2vec arrays.** The source 0908 module describes refreshed
recipes, not recovered undocumented legacy recipes. Its 50-Hz Mel has an extra
final frame for some stories. Wav2vec checkpoint, chunking and temporal geometry
may differ. No feature array is replaced, resampled or silently truncated by
this code installation. A mismatch is a version/recipe difference to investigate,
not permission to overwrite references. BERT uses the separately named imported
BERT code; it is not produced by this driver.

## Run outside the release

Use Python 3.10 and a separate environment with the pinned
`feature_extraction_v2_requirements.txt`. Brian2Hears requires Cython and a C
compiler. The wav2vec checkpoint must be supplied explicitly; no code downloads
model weights automatically. All commands should use `python -B` to avoid
creating bytecode files inside the release.

From this directory, an example without a GPU/model checkpoint is:

```bash
python -B extract_features.py --features envelope word2vec mel \
  --stories 1 --output-root ../../../../feature_outputs_smoke_01
python -B extract_features.py --features word2vec --stories 1 \
  --output-root ../../../../word2vec_reproduction_01 --verify
python -B extract_chinese_wav2vec.py --model-dir /external/checkpoint \
  --stories 1 --device cuda:0 --output-root ../../../../wav2vec_outputs_01
python -B -m unittest -v test_mel test_chinese_wav2vec test_safety
python -B verify_features.py --report ../../../../feature_structure_check_01.json
```

Replace example external paths with user-owned locations outside the release.
Outputs must be a **new** directory, outside both the selected release and the
package containing the scripts. Existing output directories are rejected. The
standalone Mel and Chinese wav2vec scripts obey the same rule. Audio is read
from `stimuli/audio/story-NN.mp3`; static-word timings are read from
`derivatives/stimulus_annotations/word_boundaries/legacy_mat` without modification.
These timings are the existing legacy version, not a new independent alignment.

The shared driver generates four feature families; `--model-dir` is required
when `wav2vec` is selected. Default features include all four families. Supply
`--features` for a subset. The default lexicon is the unchanged sibling
`word2vec_lexicon.npz`; its frozen SHA-256 is enforced.

`--verify` compares only available same-rate references in the 0909 layout.
It reads references only after generation. Newly generated 100-Hz Mel/wav2vec
and timestamp files have no installed counterpart and are labelled
`NOT_COMPARABLE`. Any differing comparable file (or no comparable file) causes a
nonzero exit after outputs and a complete manifest are saved. Hash equality is
reported only for actual identical files; no interpolation or endpoint clipping
is used to manufacture agreement. The structural verifier records current
hashes and finite/shape checks; it is not a historical reproduction test.

## Frozen wav2vec model files

The three required external files and SHA-256 checksums are:

| File | SHA-256 |
|---|---|
| `pytorch_model.bin` | `c8a5554a79c3bbbe76f2e43d3d4b4369c8c2abd5515e623192e0381d7e5e7b3f` |
| `config.json` | `53ba47fee1b3630c489e2525af7102c74f05d23dbdd49b9265ff809444c0eabb` |
| `preprocessor_config.json` | `d325e3677f9bdbd1086f9f1eccae922b82c971b8a250085248452df1ac621701` |

Audio is decoded/resampled by librosa to mono 16 kHz (`soxr_hq`). Three-second
strides include 100 extra right-context samples; tails shorter than 500 samples
are skipped. Chunks are individually input-normalized, outputs concatenated,
then FFT-resampled with float64 working precision and float32 output. Inference
uses eager attention, FP32, deterministic algorithms, seed 20260909 and disabled
TF32. The 100-Hz result is interpolated, not native 10-ms model output. No PCA is
performed. Identical results across different GPUs/libraries are not guaranteed.

For Mel, the frame count is `ceil(decoded_frames * 100 / decoded_rate)` and
centers are `k/rate`, strictly before source duration. BLAS is limited to one
thread to preserve the source recipe's summation order. Extraction manifests
record audio/model/code/output hashes and software versions. Consult
`FEATURE_EXTRACTION_V2_NOTICES.md` for third-party notices.
