# Datasets

## Status

No evaluation dataset has been downloaded or integrated yet. The only image in the
repository is a single CC0 sample used for smoke tests and the README example.

| Asset | Purpose | Licence (verified) | Status |
|---|---|---|---|
| `data/samples/amsterdam_shopping_street_cc0.jpg` | Smoke test, README example | CC0 1.0, Wikimedia Commons (read 2026-10-28) | In repository |
| COCO 2017 (validation split) | Detector quality metrics (planned) | Annotations CC BY 4.0; images carry varied Flickr licences | **Pending.** Not downloaded. Image-level licences must be reviewed before any redistribution. |
| BDD100K / Cityscapes | Driving and urban scenes for obstacle alerts (candidates) | Not yet checked | **Pending.** Licence terms, including non-commercial clauses, must be read before use. |

## Protocol requirements before any evaluation

1. Record the dataset name, version, split identifiers and download URL.
2. Read the dataset licence and any image-level restrictions. Record the verification date.
3. Keep train, validation and test splits separate. Keep whole video sequences in one
   split to avoid near-duplicate frames leaking across splits.
4. Do not tune thresholds on the test split. Tune on validation only.
5. Never copy a dataset into Git. Store it outside the repository and reference it by path
   in a local, ignored configuration.

No data-leakage check has been performed because no split exists yet.
