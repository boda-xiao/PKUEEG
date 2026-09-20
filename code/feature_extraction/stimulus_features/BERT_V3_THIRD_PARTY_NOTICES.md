# BERT v3 third-party model notice

BERT-base-Chinese is by Google and its model source declares Apache 2.0:
https://huggingface.co/google-bert/bert-base-chinese

The frozen checkpoint revision is
`8f23c25b06e129b6c986331a13d8d025a92cf0ea`. Model file fingerprints are in
`bert_v3_model_checksums.json`; the original license text copied from the 0908
source package is `BERT_V3_APACHE-2.0.txt`. Model weights are not distributed by
this additive code import.

Reference: Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova (2019),
*BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding*, https://aclanthology.org/N19-1423/ .

This import contains BERT feature extraction and rasterization only. It does not
include MFA model files, alignment software or newly generated alignments.
Third-party licenses take precedence over a dataset CC0 notice for the relevant
assets. This notice does not assert that a model software license automatically
applies to every model-output feature or resolves stimulus source permissions.

