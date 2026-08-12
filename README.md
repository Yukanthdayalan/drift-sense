# Drift-Sense: Precision Wafer Localization Engine

An advanced, scale-invariant image registration engine designed for the semiconductor inspection localization challenge. Drift-Sense is engineered to localize extremely periodic and highly distorted sub-pitch semiconductor structures (FinFETs) inside large search areas, without relying on external metadata.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Tests](https://img.shields.io/badge/tests-203%20passed-success)
![Accuracy](https://img.shields.io/badge/accuracy%20(â‰¤5px)-100%25-success)

## Overview

In semiconductor wafer inspection, matching a small template (reference) against a large Field of View (search image) is challenging due to sub-pitch periodic structures, scale variations, and intense manufacturing/sensor noise. Drift-Sense tackles this via a deterministic, classical computer vision pipeline optimized for periodic ambiguity resolution, completely avoiding black-box neural networks.

## Key Capabilities

- **Zero-Metadata Inference:** Computes exact geometric mapping using only image data.
- **Top-2 Sharpness Verification:** Resolves sub-pitch aliasing by dynamically analyzing local peak sharpness when multiple structural periods exhibit identical ZNCC intensities.
- **Scale-Invariant:** Multi-scale hierarchical search correctly locks onto patterns regardless of scaling drifts.
- **Sub-Pixel Precision:** Refines bounding boxes via localized quadratic/Gaussian interpolation for sub-pixel accuracy.
- **Production-Ready Performance:** Average CPU runtime of ~2.7s per megapixel search.

## How the Algorithm Works

1. **Preprocessing:** Reference and search images are normalized for illumination invariance.
2. **Multi-Scale Search:** `estimate_top_n_scales` identifies the top 2 scale candidates across a bounded scale space.
3. **ZNCC Matching:** `compute_zncc_fft` executes an ultra-fast FFT-based Zero-mean Normalized Cross-Correlation to find candidate regions.
4. **Candidate Peak Detection:** `detect_best_peak` extracts structural coordinates with Non-Maximum Suppression (NMS) and a center-aware spatial tie-breaker.
5. **Periodic-Structure Disambiguation:** If the intensity gap between the top 2 candidates is $< 0.015$, the engine triggers **Sharpness Verification** (`get_sharpness`), deferring to the mathematically sharper peak to evade sub-pitch aliasing.
6. **Sub-Pixel Refinement:** `refine_subpixel` fits a 2D polynomial surface around the discrete pixel peak to resolve continuous coordinates.
7. **Coordinate Translation:** Outputs the precise $(x, y)$ center of the target in search-image coordinates.

## Installation

A clean virtual environment is highly recommended. Only minimal production dependencies are required.

```bash
# Clone the repository
git clone https://github.com/Yukanthdayalan/drift-sense
cd drift-sense

# Create a clean virtual environment
python -m venv .venv

# Activate the environment (Windows PowerShell)
.\.venv\Scripts\activate
# If restricted by execution policies, use explicitly: .\.venv\Scripts\python.exe

# Install production dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies for testing
pip install -r requirements-dev.txt
```

## Production Inference

The inference engine runs completely standalone. It suppresses all internal verbose logging to maintain a clean machine-readable pipe.

```bash
python inference.py <path_to_reference.png> <path_to_search.png>
```

### CLI Output

The mandatory script outputs exactly one line to `stdout`:
```text
(174.8601, 366.7831)
```

Optionally, you can dump a structured JSON output by providing the `--output` flag (the single line to `stdout` is preserved):
```bash
python inference.py <reference.png> <search.png> --output output.json
```
**`output.json`**
```json
{
  "prediction_x": 174.860124,
  "prediction_y": 366.783182
}
```

## Dataset Generation

You can procedurally generate completely unobserved synthetic periodic defect datasets (FinFET style) for blind evaluation.

```bash
python generate_dataset.py --architecture finfet --num-pairs 30 --output-dir evaluation_dataset
```
This generates pairs of `reference.png` and `search.png`. The absolute ground-truth center coordinate is recorded in each sample's `ground_truth.json`. **Note:** The inference engine never touches this ground truth.

## Testing

The project maintains a comprehensive pytest suite covering the inference orchestrator, subpixel optimization, and geometric disambiguation modules.

```bash
python -m pytest -q
```
*(Current status: 203 passing)*

## Evaluation Results

The production algorithm was independently evaluated on a 50-sample high-noise synthetic FinFET dataset (`results/summary.json`).

| Metric | Result |
|---|---:|
| Samples | 50 |
| â‰¤1 px accuracy | 52.0% |
| â‰¤2 px accuracy | 88.0% |
| **â‰¤5 px accuracy** | **100.0%** |
| Mean error | 1.085 px |
| Median error | 0.987 px |
| Maximum error | 3.076 px |
| Mean runtime | 2.766 s/sample |
| Median runtime | 2.725 s/sample |
| Maximum runtime | 3.404 s/sample |

*100% of evaluated samples were within the 5-pixel tolerance limit. The maximum registered error was ~3.076 pixels, showcasing total elimination of catastrophic >10px scale aliasing.*

## Repository Structure

```text
drift-sense/
â”œâ”€â”€ generate_dataset.py     # CLI interface for synthetic FinFET datasets
â”œâ”€â”€ inference.py            # Production standalone evaluator entry point
â”œâ”€â”€ requirements.txt        # Production dependencies (numpy, opencv-python, scipy)
â”œâ”€â”€ requirements-dev.txt    # Development dependencies (pytest)
â”œâ”€â”€ README.md               # Documentation
â”œâ”€â”€ SUBMISSION_CHECKLIST.md # Official submission verification log
â”œâ”€â”€ src/drift_sense/        # Core algorithm package
â”‚   â”œâ”€â”€ dataset.py          # Synthetic generation logic
â”‚   â”œâ”€â”€ matcher.py          # End-to-end inference orchestrator
â”‚   â”œâ”€â”€ peak_detector.py    # NMS and sharpness disambiguation
â”‚   â”œâ”€â”€ scale_search.py     # Multi-scale Top-N frequency estimation
â”‚   â”œâ”€â”€ subpixel.py         # Sub-pixel quadratic interpolation
â”‚   â””â”€â”€ zncc.py             # FFT-based template matching
â”œâ”€â”€ docs/                   # Auxiliary documentation
â”œâ”€â”€ references/             # Literature/Citation placeholders
â”œâ”€â”€ results/                # CSV and JSON evaluation exports
â””â”€â”€ tests/                  # Pytest unit and integration suite
```

## Reproducibility

Any user can immediately clone the repository, install `requirements.txt`, run `generate_dataset.py`, and validate the localization pipeline using `inference.py`. No GPU, proprietary framework, or hidden calibration metadata is required.

## Limitations

- **DRAM Support:** Full procedural generation of generalized DRAM topology is not currently active (defaults to general periodic structures mapping similarly to FinFET).
- **Scale Bounds:** The system heavily optimizes for the `[9.5, 10.5]` scaling bracket expected in standard inspection inputs. Drastic scaling beyond this space may require modifying the internal `config.scale_search` ranges.
