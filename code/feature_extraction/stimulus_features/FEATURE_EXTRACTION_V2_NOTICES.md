# Imported acoustic extraction model notices

This file preserves the applicable model attribution from the 0908 source
module. Importing extraction code does not change the license of a model or
relicense existing dataset files.

## Chinese wav2vec 2.0

The source extractor uses the frozen `TencentGameMate/chinese-wav2vec2-large`
checkpoint. The source module records that its upstream model card declares
the MIT license and attributes pretraining to the WenetSpeech L subset.

Upstream: https://huggingface.co/TencentGameMate/chinese-wav2vec2-large

The checkpoint is NOT included by this import and is never downloaded by the
extractor. Obtain the upstream checkpoint and its license separately; three
required files are SHA-256 pinned in `extract_chinese_wav2vec.py`. Code presence
alone does not establish that the legacy 0909 wav2vec arrays used this model.

## Static word vectors

The already installed `word2vec_lexicon.npz`, word2vec notices, and word2vec
extractor are reused unchanged. The lexical subset derives from fastText Chinese
Common Crawl/Wikipedia `cc.zh.300.bin`, including the recovered character-average
fallback. These vectors and derived arrays retain CC BY-SA 3.0 terms, not CC0.

Source: https://fasttext.cc/docs/en/crawl-vectors.html

License: https://creativecommons.org/licenses/by-sa/3.0/

Attribution: Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin and
Tomas Mikolov, *Learning Word Vectors for 157 Languages*, 2018.

No model weights, duplicate lexical subsets, new features or results are added
by this acoustic-code import.
