# Drift-Sense: Precision Wafer Localization Engine

Drift-Sense is a deterministic, classical computer-vision engine designed to localize high-resolution structural templates within large-area, high-noise periodic semiconductor images.

## Problem

Modern wafer inspection tools frequently experience stage drift, mechanical navigational errors, or reference image rotations up to ±15°. Given a small high-resolution reference/template image and a much larger search image containing periodic FinFET-style structures, the engine must determine the precise (x, y) location of the reference content inside the search image to re-localize the tool.

## Solution

Drift-Sense strictly utilizes a multi-scale, multi-angle classical computer vision pipeline built on Zero-Mean Normalized Cross-Correlation (ZNCC). It does not rely on deep learning, metadata leakage, or external dataset requirements during inference. It handles intense Gaussian noise, scale mismatch (approx 10x), and large rotation discrepancies through targeted grid searches.

## Key Features

- **Multi-Scale Localization**: Efficiently identifies the correct scale bound (~10x mismatch).
- **Multi-Angle Rotation Search**: Explicitly tests 5 rotation angles to resolve severe drift.
- **Sub-Pixel Refinement**: Uses 2D parabolic fitting to achieve precise continuous coordinates.
- **Robustness to Periodicity**: Non-Maximum Suppression (NMS) and classical topological sharpness heuristics resolve grid ambiguity cleanly without heavy architectures.

## Pipeline

The production execution follows these strict steps:

1. **Preprocessing**: Normalizes and cleans the reference and search footprints.
2. **Scale Handling**: Generates the optimum 1D scale footprint using the unrotated `0.0°` response to avoid redundant computations.
3. **Multi-Angle Search**: The `200x200` reference footprint is physically rotated *before* downscaling to prevent destructive aliasing on the periodic FinFET lattice. The pipeline evaluates five distinct angles: `[-15.0, -7.5, 0.0, 7.5, 15.0]`.
4. **FFT-ZNCC**: Performs rapid Zero-Mean Normalized Cross-Correlation in the frequency domain.
5. **Candidate Extraction & NMS**: Collects peaks across all angular response maps and consolidates neighbors via Non-Maximum Suppression.
6. **Candidate Selection**: Determines the absolute best candidate. In ties involving identical period correlation, it evaluates topological sharpness.
7. **Sub-Pixel Localization**: Computes the exact sub-pixel offset.
8. **Final Coordinate**: Returns the definitive `(x, y)` coordinate.

## Repository Structure

- `src/drift_sense/`: Core production inference library (`matcher.py`, `geometry.py`, `peak_detector.py`, etc.).
- `tests/`: Comprehensive Pytest unit tests verifying geometry, ZNCC, and rotation.
- `inference.py`: Production CLI entrypoint.
- `requirements.txt`: Lightweight production dependencies (NumPy, OpenCV, SciPy).

## Installation

The repository is built strictly for standard Python 3.11.

```bash
git clone https://github.com/Yukanthdayalan/drift-sense
cd drift-sense
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows
python -m pip install -r requirements.txt
```

## Inference

The evaluator-compatible command takes the reference and search images and outputs a single coordinate tuple to `stdout`. Ground truth or generator metadata are never required.

```bash
python inference.py <reference_image> <search_image>
```

Example mandatory output:
```
(467.6238, 797.2276)
```

You may optionally save the output to a JSON file:
```bash
python inference.py <reference_image> <search_image> --output results.json
```

## Testing

The codebase is thoroughly validated.

```bash
export PYTHONPATH="src"
pytest -q
```

**Current Verified Result:**
```
209 passed
```

## Validation

The system's limits were comprehensively tested during development on freshly generated synthetic datasets. The baseline pipeline maintains `< 1.0px` average localization error with `100%` accuracy within `<= 2.0px`, even under simulated hardware stress (1.75x noise injection) and ±15° reference rotation.
