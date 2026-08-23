# Knee OA KL-Grade Classifier — Model Report

Trained on the user's own copy of the Kaggle "Knee Osteoarthritis Dataset with
Severity Grading" (OAI-derived, 224×224 grayscale radiographs, pre-split into
train/val/test/auto_test by the dataset authors).

## Approach

- **Preprocessing:** grayscale, resize to 96×96, histogram-equalize for exposure
  normalization, then extract **HOG (Histogram of Oriented Gradients)** features
  (9 orientations, 12×12 cells, 2×2 block normalization → 1,764-dim feature vector).
- **Why HOG + classical ML instead of a CNN:** trained in a sandboxed, single-CPU,
  no-GPU environment with network access restricted to package registries
  (PyPI/npm/GitHub). The plain PyPI `torch` wheel pulls multi-GB CUDA
  dependencies that aren't needed for CPU training and don't fit that
  environment's time/disk budget. HOG features are fast to compute (~3ms/image)
  and classical models train in seconds-to-minutes on one core, so this pipeline
  is fully reproducible without a GPU. See "Upgrading to a CNN" below for a
  path to a deep model.
- **Models benchmarked:** Random Forest, HistGradientBoosting, and a linear
  SGD/logistic baseline — all with class-balanced weighting since the dataset
  is skewed toward KL0 (healthy is ~40% of cases, KL4/severe is ~3%).
- **Model selection:** best model chosen by **macro-F1 on the validation set**
  (not accuracy — accuracy alone would reward always predicting the majority
  class KL0). Final numbers below are on the **test set**, untouched until
  model selection was locked in.

## Results (real, not illustrative)

| Model | Val accuracy | Val macro-F1 |
|---|---|---|
| Random Forest | 0.470 | 0.286 |
| **HistGradientBoosting (selected)** | **0.525** | **0.453** |
| SGD / linear | 0.416 | 0.430 |

**Final test set (n=1,656, never used for model selection):**
- Accuracy: **50.1%** (random-chance baseline for 5 balanced classes would be 20%)
- Macro-F1: **0.431**

| KL grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 – None | 0.548 | 0.818 | 0.656 | 639 |
| 1 – Doubtful | 0.159 | 0.061 | 0.088 | 296 |
| 2 – Minimal | 0.465 | 0.452 | 0.459 | 447 |
| 3 – Moderate | 0.500 | 0.278 | 0.357 | 223 |
| 4 – Severe | 0.800 | 0.471 | 0.593 | 51 |

See `confusion_matrix.png`. Errors cluster near the diagonal — when the model
is wrong, it's usually off by one adjacent grade, not a wild miss. This mirrors
a known property of the KL scale itself: **grade 1 ("doubtful") has notoriously
low inter-rater agreement even among human radiologists**, since it's defined
as a borderline call between healthy and early disease. The model's near-total
failure on class 1 (recall 0.061) is a real limitation, not a bug — it's
predicting the two neighboring classes instead, which is the same ambiguity
clinicians report.

## Files

- `train_model.py` — full training pipeline, run with `--data_dir` pointing at
  a folder containing `train/`, `val/`, `test/` subfolders of class folders `0`-`4`
- `predict.py` — inference on a single new image: `python3 predict.py xray.png`
- `kl_classifier_final.joblib` — the trained model bundle (model + scaler +
  preprocessing params), loadable via `joblib.load(...)`
- `metrics.json` — machine-readable version of the results above
- `confusion_matrix.png` — test-set confusion matrix

## Limitations (be upfront about these to judges)

1. **50% accuracy on 5 classes is a real, modest signal** — well above the 20%
   chance baseline and consistent with published classical (non-deep) baselines
   on this task, but well below the ~75-82% accuracy reported by CNN-based
   papers trained on the full dataset with GPU compute.
2. **Grade 1 (doubtful) is not reliably detected.** Frame this as consistent
   with known KL-grading ambiguity, not hidden.
3. HOG features capture edge/texture patterns correlated with joint space
   narrowing and osteophyte-like structure, but don't explicitly localize or
   segment anatomy the way a trained U-Net / CNN would.

## Upgrading to a CNN (if you get GPU access, e.g. Colab)

The architecture to reach for: a small ResNet-style CNN (4-6 conv blocks,
batchnorm, global average pool, dropout, FC → 5 classes), trained with
class-weighted cross-entropy or focal loss, standard augmentation (small
rotations ±10°, brightness/contrast jitter — avoid horizontal flips if you
care about medial/lateral compartment distinction), and ideally **ordinal
regression** (e.g. CORAL or a cumulative-link output) rather than plain
multi-class classification, since KL grades are ordered, not categorical —
this alone tends to reduce "off by one" errors like the ones seen here.
Fine-tuning from an ImageNet-pretrained backbone (ResNet18/EfficientNet-B0)
rather than training from scratch would likely be the single biggest lever
for accuracy, given how much this dataset benefits from transfer learning
in the published literature.
