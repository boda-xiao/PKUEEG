# Integration scope (2026-09-19)

Source: `PKUEEG_release_candidate_20260908/code/speech_retrieval`.
Target: `PKUEEG_release_20260909/code/speech_retrieval`.

Only the copied code, configuration, documentation, and tests were adapted.
Current input EEG and features are reused read-only; no historical model,
result, log, data array, or obsolete source feature/checksum manifest is installed.

Changes: public BIDS-like EEG paths; current feature directories; native 50-Hz
wav2vec declared correctly and polyphase-resampled to the existing 100-Hz
analysis grid in memory; unavailable word2vec explicitly disabled; case-insensitive
channel validation; external output guards; updated portability tests and README.
Ridge, splitting, lag, scoring and group-SEM algorithms retain the source defaults.

Default input selection is now envelope + wav2vec. The current release differs
from the 0908 candidate/repair data and a numerical equivalence claim would be
incorrect. No full training rerun is part of this integration. Consult README.md
for commands, inputs, outputs, and limitations.
