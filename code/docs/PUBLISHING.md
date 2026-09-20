# Publishing scope

OpenNeuro hosts the BIDS data; [boda-xiao/PKUEEG](https://github.com/boda-xiao/PKUEEG)
hosts this code-only companion. The initial publication is based on the local
20260909 code with later additions, not a claim that every file is already in
the published [1.0.0 snapshot](https://openneuro.org/datasets/ds008834/versions/1.0.0).

The upload includes source code, documentation, configurations, tests, small
model fingerprints, and a licensed frozen lexical subset. It excludes raw and
preprocessed EEG, audio/text, time-series feature arrays, behavioral data,
model weights, old results, private checks, caches, credentials and environments.
`.bidsignore` is not an upload filter; the Git staged file list must be reviewed.

Current instructions are maintained in:

- [Repository README](../../README.md)
- [Third-party licensing scope](../../THIRD_PARTY_NOTICES.md)
- [OpenNeuro linking text and Draft/version instructions](../../docs/OPENNEURO_LINKING.md)
- [Actual publication checks](../../docs/PUBLICATION_CHECKS.md)

The original server release remains unchanged. OpenNeuro reverse-link text is
prepared for owner review; publishing this Git repository does not automatically
edit the online Draft or create a new snapshot. Keep dataset and code versions
distinct. Do not assert a code DOI before a real archive assigns it.

Raw-BIDS validation, numerical integrity, alignment accuracy and scientific
reproducibility are separate checks. No new ethics approval or blanket license
for third-party models/stimuli is implied by code publication.
