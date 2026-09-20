# Licensing scope and third-party resources

The original PKUEEG release's CC0 declaration is retained in `LICENSE` for
PKUEEG-authored material covered by that declaration. It is not a blanket
relicensing of third-party resources. Existing attributions and notices are
retained with the imported code.

## Included fastText-derived lexical subset: CC BY-SA 3.0

`code/feature_extraction/stimulus_features/word2vec_lexicon.npz` is a frozen
vocabulary subset of the Chinese `cc.zh.300.bin` model. It contains model-derived
vectors, not executable source code. Its upstream **CC BY-SA 3.0** terms remain
applicable; this file is **not CC0**. Redistribution must preserve those terms,
attribution, and the record of modifications. See the detailed
[word-vector notice](code/feature_extraction/stimulus_features/WORD2VEC_THIRD_PARTY_NOTICE.md)
and accompanying `word2vec_lexicon.json`.

- Upstream: https://fasttext.cc/docs/en/crawl-vectors.html
- License: https://creativecommons.org/licenses/by-sa/3.0/
- Attribution: Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin and
  Tomas Mikolov, *Learning Word Vectors for 157 Languages* (2018).
- Modifications: vocabulary subsetting, documented character-average fallback,
  and storage as Unicode labels and float32 vectors in NPZ format.

The historical feature name `word2vec` does not change the vectors' fastText
origin. The fastText software license and pretrained-vector license are distinct.

## External models: not redistributed

- Chinese wav2vec2: `TencentGameMate/chinese-wav2vec2-large`; consult its upstream
  model card and the retained
  [feature notices](code/feature_extraction/stimulus_features/FEATURE_EXTRACTION_V2_NOTICES.md).
- BERT: `google-bert/bert-base-chinese`; the retained
  [BERT notice](code/feature_extraction/stimulus_features/BERT_V3_THIRD_PARTY_NOTICES.md)
  and [Apache-2.0 text](code/feature_extraction/stimulus_features/BERT_V3_APACHE-2.0.txt)
  describe the external model.

Python dependencies and external tools keep their own licenses. No neural model
weights, stimulus recordings/texts, or behavioral tables are added to this
repository. Do not infer permission to redistribute separately obtained resources
from this repository's license.
