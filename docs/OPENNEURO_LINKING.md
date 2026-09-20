# Linking OpenNeuro and this GitHub repository

Data: https://openneuro.org/datasets/ds008834

Code: https://github.com/boda-xiao/PKUEEG

These are separate resources with independent version histories. This document
provides text for the dataset owner to review and apply to an OpenNeuro **Draft**.
Publishing this repository does not update OpenNeuro or the original local dataset.

## Text to add to the OpenNeuro README

```markdown
## Code availability

Processing, feature-extraction and technical-validation code is maintained at:
https://github.com/boda-xiao/PKUEEG

The GitHub repository is a code-only companion to this dataset. Its README
documents dependencies, external resources, known limitations and version
compatibility. Code additions do not retroactively change earlier OpenNeuro
snapshots. Cite the dataset version actually analyzed and the exact code
commit or code release used for the analysis.

Behavioral accuracy tables and an executable MFA-v3 alignment workflow are not
included in the initial companion-code publication. Model weights must be
obtained separately under their respective terms. Inclusion of extraction
code does not imply exact reconstruction of every stored legacy feature.
```

If the dataset README still says no public repository URL has been assigned,
replace that obsolete statement. Leave unrelated scientific metadata unchanged.

## Dataset-description edit

Append this URL to the existing `ReferencesAndLinks` list in
`dataset_description.json`, preserving all existing values:

```text
https://github.com/boda-xiao/PKUEEG
```

For the locally inspected metadata, the merged field would be:

```json
"ReferencesAndLinks": [
  "https://www.gstudios.com.cn/",
  "https://github.com/boda-xiao/PKUEEG"
]
```

This is a **field fragment**, not a replacement dataset-description file. Recheck
the current online Draft before merging because it may contain more links. Do
not replace `DatasetDOI` with a code DOI. `CodeURL`, if used, belongs inside an
appropriate `GeneratedBy` object and should identify the code that actually
generated those data, not merely an unrelated downstream-analysis repository.

## Proposed changelog entry

If 1.0.0 remains the newest version and no data changes are pending, a metadata-only
Patch release such as 1.0.1 is appropriate to consider. Suggested changelog text:

```text
Added the PKUEEG companion-code repository link and clarified code availability
and version compatibility. EEG, stimuli and derived feature arrays are unchanged.
```

Use the final sentence only after verifying that the Draft contains no other
pending changes. A new snapshot includes all Draft changes. Updating a Draft
alone does not make its changes publicly visible or alter the immutable 1.0.0
snapshot. No new snapshot is created by any script in this repository.

## Final check

1. Open the public GitHub README while logged out and follow its OpenNeuro link.
2. After the owner publishes the new OpenNeuro snapshot, follow its code link.
3. Record the public dataset version and tested code commit in analysis outputs.
4. If a code DOI is later obtained through archival, cite it separately from the
   dataset DOI. Do not invent either DOI from a naming convention.

Official guidance:

- [OpenNeuro dataset management](https://docs.openneuro.org/user_guide.html#managing-your-dataset)
- [BIDS dataset description](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/dataset-description.html)
