# Third-party model notice

The static word vectors in `word2vec_lexicon.npz` (named
`assets/word_lexicon.npz` in the archived source package) originate from the fastText
Chinese Common Crawl/Wikipedia model `cc.zh.300.bin`. The lexical subset and
character-average fallback are described in the accompanying model metadata.
The subset was regenerated from that independent pretrained model using the
released word labels, not extracted from any time-major target feature array.

The upstream vectors are distributed under **CC BY-SA 3.0**, not CC0:

- Source and model information: https://fasttext.cc/docs/en/crawl-vectors.html
- License: https://creativecommons.org/licenses/by-sa/3.0/
- Attribution: Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin and
  Tomas Mikolov, *Learning Word Vectors for 157 Languages*, 2018.

The lexical subset retains the upstream CC BY-SA 3.0 terms. Changes consist of
selecting the released vocabulary, applying the recovered character-average
fallback for zero vectors, and storing Unicode labels plus float32 vectors in
NPZ format. The existing release-level CC0 statement does not change these
third-party model terms. This note does not revise or resolve the licensing of
other pre-existing release files; review that scope before public distribution.

The software package fastText and its pretrained vectors are distinct resources;
the library's license must not be substituted for the vectors' license.
