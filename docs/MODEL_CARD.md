# Model card: VisIA Spatial Intelligence (research prototype)

## Model details

This is a pipeline, not a single trained model. It combines pretrained components:

| Role | Checkpoint | Licence | Pinned commit |
|---|---|---|---|
| Object detection | `facebook/detr-resnet-50` (COCO-trained) | Apache-2.0 | `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` |
| Relative depth | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| Tracking | Hand-written IoU tracker (no learned parameters) | Project licence | — |
| Spatial reasoning | Hand-written rules with fixed thresholds | Project licence | — |

No component has been trained or fine-tuned in this project.

## Intended use

* Research and education on how monocular perception can be combined with transparent
  spatial rules.
* Demonstrating a reproducible CV pipeline, its configuration and its limitations.

## Out-of-scope and prohibited use

* Guiding, warning or protecting people who are blind or have low vision, or anyone else
  who relies on the output to avoid harm.
* Driving, traffic control, autonomous vehicles or robots that move near people.
* Surveillance or identifying individuals.
* Any use that depends on the alerts being complete or correct.

## Training data provenance

The detector's training data is COCO 2017. The COCO annotations are CC BY 4.0. The images
come from Flickr with varied licences (see `LICENSE_AUDIT.md`). The depth model's training
data is described in the upstream Depth Anything V2 paper and was not reviewed in this
audit. This project performs no training.

## Evaluation

No accuracy evaluation has been performed. Only a runtime benchmark on one image has been
run; see `EVALUATION.md`. Do not cite a detection or alert accuracy for this project.

## Known failure modes

See `LIMITATIONS.md`. In short: relative depth is uncalibrated, the corridor is a fixed
assumption, risk levels are heuristic, and COCO does not cover many real obstacles.

## Ethical considerations

Pedestrians and bystanders appear in photos. The pipeline does not store identities and
does not perform face recognition. Outputs can contain bounding boxes around people, so
treat stored results as personal data where they show identifiable people.
