# Code changelog

## Initial companion-code publication - 2026-09-20

- Import the inspected local 20260909 release code, including later 20260919
  technical-validation and feature-extraction additions, without changing its
  scientific Python implementations or default experiment hyperparameters.
- Retain the existing `code/` layout, tests, provenance, lexical asset, model
  fingerprints, and component-specific notices.
- Add English repository documentation, data-download links, explicit-path
  run examples, a code-only upload policy, and OpenNeuro reverse-link text.
- Add an external retrieval-config helper and synthetic safety tests; word2vec
  remains opt-in and requires actual files.
- Clarify later local word2vec availability, missing executable MFA workflow,
  external models/behavioral input, and limitations of reproducibility claims.
- Retain the source CC0 declaration with explicit third-party asset exceptions.

This code publication does not upload data, alter the original local release,
change OpenNeuro 1.0.0, or rerun full scientific experiments. See
`docs/PUBLICATION_CHECKS.md` for the actual bounded checks performed.
