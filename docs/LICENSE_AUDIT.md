# Licence audit

Audit date: **2026-10-28**. Auditor: project maintainer (VUBOH BENOIT), by reading the
licence text or metadata cited in each row. Each row was checked before the component
was integrated.

Status key: **Verified** = licence text or model card read directly, and the source
pinned. **Verified (metadata)** = licence read from the installed package metadata.
**Unresolved** = a usage question remains open; see notes.

## Integrated components

| Component | Version / revision | Licence | Source (verified) | Commercial use | Redistribution restrictions | Status |
|---|---|---|---|---|---|---|
| `facebook/detr-resnet-50` weights | commit `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`; `model.safetensors` SHA-256 `830f5e2e…420ad81` | Apache-2.0 (model card `license: apache-2.0`) | https://huggingface.co/facebook/detr-resnet-50 | Permitted under Apache-2.0 | Keep licence and NOTICE; state changes | **Verified**, with an unresolved COCO data-provenance note (below) |
| DETR architecture (via `transformers`) | transformers 4.49.0 | Apache-2.0 | installed package metadata; https://github.com/facebookresearch/detr LICENSE (Apache-2.0, read 2026-10-28) | Permitted | Apache-2.0 terms | Verified (metadata) |
| `depth-anything/Depth-Anything-V2-Small-hf` weights | commit `5426e4f0f36572d16453bbda7a8389317b1bef99`; `model.safetensors` SHA-256 `3152477c…a16b70` | Apache-2.0. The upstream README states: "Depth-Anything-V2-Small model is under the Apache-2.0 license." | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf ; https://github.com/DepthAnything/Depth-Anything-V2 README (read 2026-10-28) | Permitted | Apache-2.0 terms | **Verified** |
| Depth Anything V2 code | upstream repository, `main` | Apache-2.0 (LICENSE read 2026-10-28) | https://github.com/DepthAnything/Depth-Anything-V2 | Permitted | Apache-2.0 terms | Verified (code is used via `transformers` only) |
| `transformers` | 4.49.0 | Apache-2.0 | installed metadata | Permitted | Apache-2.0 terms | Verified (metadata) |
| `timm` | 1.0.30 | Apache-2.0 | installed metadata; PyPI 1.0.22 metadata checked | Permitted | Apache-2.0 terms | Verified (metadata) |
| `torch` | 2.2.2 (CPU wheel on Linux) | BSD-3-Clause | installed metadata | Permitted | BSD-3 notice | Verified (metadata) |
| `torchvision` | 0.17.2 | BSD-3-Clause | installed metadata; PyPI metadata checked | Permitted | BSD-3 notice | Verified (metadata) |
| `numpy` | 1.26.4 | BSD-3-Clause (wheels bundle third-party runtime libraries, whose notices are included in the wheel) | installed metadata | Permitted | BSD-3 notice; bundled-library notices | Verified (metadata) |
| `Pillow` | 13.0.0 | MIT-CMU (HPND) | installed metadata | Permitted | Notice | Verified (metadata) |
| `PyYAML` | 6.0.3 | MIT | installed metadata | Permitted | Notice | Verified (metadata) |
| `huggingface-hub`, `safetensors`, `tokenizers`, `requests` | transitive | Apache-2.0 | installed metadata | Permitted | Apache-2.0 terms | Verified (metadata) |
| `tqdm` | transitive | MPL-2.0 AND MIT | installed metadata | Permitted | MPL-2.0 is file-level copyleft. We do not modify its files, so no obligation applies to our code. | Verified (metadata) |
| `pytest`, `ruff` | dev only | MIT | installed metadata | Permitted | Dev tooling only; not shipped | Verified (metadata) |
| Sample image `data/samples/amsterdam_shopping_street_cc0.jpg` | Wikimedia Commons file "2023 Amsterdam city - People are walking in the shopping street Ferdinand Bolstraat…", author FotoDutch; 1280×960 JPEG thumbnail of the original TIFF | **CC0 1.0** (Commons `LicenseShortName` and `LicenseUrl` read 2026-10-28) | https://commons.wikimedia.org/wiki/File:2023_Amsterdam_city_-_People_are_walking_in_the_shopping_street_Ferdinand_Bolstraat_by_a_car_of_municipal_cleaning;_free_download_photo_by_Fons_Heijnbroek,_street_photography_The_Netherlands.tif | Permitted | None (CC0). Attribution is not required and is given anyway. | Verified |

## Rejected components

| Component | Reason | Decision |
|---|---|---|
| Ultralytics YOLO (any version) and YOLO-derived checkpoints | AGPL-3.0 licence, prohibited by the project constraints | **Rejected** |
| `depth-anything/Depth-Anything-V2-Base-hf`, `-Large-hf`, `-Giant` | Checkpoint licence `cc-by-nc-4.0` (non-commercial); the upstream README states that Base, Large and Giant are CC-BY-NC-4.0 | **Rejected** for commercial use |
| `nvidia/segformer-b0-finetuned-ade-512-512` | Hugging Face tag `license:other`; the full terms were not established. Its licence and dataset provenance (ADE20K) are unresolved. | **Not integrated.** Walkable-area segmentation is therefore not implemented. |

## Candidates evaluated but not integrated

| Component | Licence (metadata) | Status |
|---|---|---|
| `PekingU/rtdetr_r50vd` | Apache-2.0 tag | Candidate for a faster detector. Not evaluated, so not integrated. |

## Unresolved uncertainties

1. **COCO provenance of DETR weights.** `facebook/detr-resnet-50` was trained on COCO 2017.
   COCO annotations are CC BY 4.0. The COCO images come from Flickr with varied
   per-image licences. We did not download COCO, and the checkpoint's Apache-2.0 licence
   is taken as stated by its publisher. Commercial users should decide for themselves
   whether that is acceptable for their use.
2. **Licence of model outputs.** Detections and depth maps are derived outputs. This
   audit does not establish any additional restriction on them beyond the checkpoint
   licences above.
3. **Project licence.** The project code is released under Apache-2.0 (see `LICENSE`),
   chosen to match the Apache-2.0 components it uses. This is a maintainer decision.

## Datasets

No dataset has been downloaded or integrated. Evaluation datasets are pending; see
[DATASETS.md](DATASETS.md) for the planned protocol and licence checks required before use.
