# Limitations

VisIA Spatial Intelligence is a research and portfolio prototype. It is **not** a
certified mobility aid, navigation system or safety system. It does not guarantee
pedestrian safety and must not be used to guide a person who cannot see the environment
or to avoid traffic.

## Perception

* **Relative depth only.** The depth model outputs relative inverse depth, normalised per
  image. Values are not metres and are not calibrated. Relative proximity is a relative
  ordering within one image and cannot be compared across images.
* **Depth is noisy on some surfaces.** Reflective, textureless or heavily backlit regions
  may produce incorrect relative depth. This has not been measured systematically yet.
* **Classes are limited to COCO.** Only the configured navigation classes are reported.
  Obstacles outside COCO (poles, open car doors, construction barriers, holes, kerbs,
  steps) are invisible to the system.
* **Detector misses and false positives are not measured.** No precision or recall is
  reported yet. Expect misses for small, distant, occluded or unusually posed people.
* **Not real-time on the development machine.** The steady-state end-to-end latency is
  several seconds per 1280×960 frame on an Intel i7 CPU (see `EVALUATION.md`). Video at
  interactive frame rates is not achieved.

## Spatial reasoning

* **The corridor is a fixed trapezoid.** It is a configurable, camera-dependent
  assumption, not an estimate of the real walkable area. It does not follow kerbs, paths or
  slopes, and it assumes the camera is level and roughly centred on the walking direction.
* **Walkable-area segmentation is not implemented.** No checkpoint has passed the licence
  audit (see `LICENSE_AUDIT.md`), so the system does not know where pavement is.
* **Risk levels are heuristic.** `low`, `medium` and `high` are ordering rules, not
  probabilities of collision. Thresholds have not been tuned or validated on labelled data.
* **Still images never alert.** Alerts require a track persisting for several consecutive
  frames, so a single photo produces assessments but no alerts. This is intentional.
* **No motion model.** Tracking uses IoU alone. Fast motion, occlusion longer than
  `max_missed_frames`, and crowds cause identity switches and new IDs (see
  `src/visia_spatial_intelligence/tracking/tracker.py`).
* **No ego-motion.** The camera is assumed to be static relative to the scene, so an
  approaching object is not distinguished from a stationary one.

## Data and generalisation

* **No measured generalisation.** The system has not been evaluated on any dataset.
  Performance on night scenes, rain, snow, unusual camera angles, low-resolution cameras
  or other countries is unknown.
* **Single sample image.** The only image used for smoke testing is one daytime street
  scene in Amsterdam.

## Engineering

* **Video is not supported yet.** The pipeline and reasoning layer accept frame sequences,
  but no video decoder is included.
* **Input limits.** Images are rejected above 10 MB or 4096 px on a side.
* **CPU only.** Apple Silicon MPS and CUDA paths are not implemented, and the development
  machine has no MPS.
* **Model downloads at first run.** The first run downloads about 270 MB of weights from
  Hugging Face. Offline use requires a populated cache.
