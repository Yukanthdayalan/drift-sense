# Drift-Sense: Precision Wafer Localization Engine

Drift-Sense is a deterministic, classical computer-vision engine designed to localize high-resolution structural templates within large-area, high-noise periodic semiconductor images.

## 1. Problem
Modern wafer inspection tools frequently experience stage drift or mechanical navigational errors. Given a small high-resolution reference/template image and a much larger search image containing periodic FinFET-style structures, the engine must determine the precise (x, y) location of the reference content inside the search image to re-localize the tool.

## 2. FinFET Scope
The current optimized implementation and configuration are strictly bounded to **FinFET** logic layouts. DRAM and other architectures are explicitly excluded from this submission.

## 3. Why Periodicity is Difficult
FinFET arrays consist of millions of perfectly identical structures. A purely intensity-based matcher will find dozens of mathematically identical correlation peaks separated by exactly one FinFET period. If the algorithm guesses the wrong peak, the tool drives to the wrong coordinate, missing the defect entirely.

## 4. Scale Mismatch
The problem features an approximate 10x physical scale mismatch.
- Reference dimensions: `20 x 20`
- Search dimensions: `1000 x 1000`
- Search Footprint in Search Image: `~ 200 x 200`
- Scale relationship: `S = f_search / f_reference ~ 10`

## 5. Noise Model
The synthetic generator injects independent Gaussian sensor noise with strict directionality to simulate realistic electron-beam conditions:
- **Reference noise sigma:** 5.0
- **Search noise sigma:** 8.0

## 6. Algorithm
The deterministic classical pipeline operates without machine learning, neural networks, or ground-truth metadata. It uses a Z-score normalization pass followed by a multi-scale FFT-ZNCC grid search.

## 7. Candidate Generation
Instead of assuming the highest correlation peak is correct, Drift-Sense uses spatial Non-Maximum Suppression (NMS) to generate multiple plausible candidates across the search grid.

## 8. Ambiguity Resolution
Candidates within 0.5% of the absolute maximum ZNCC score trigger the deterministic tie-breaker. The engine evaluates the topological `sharpness` of the correlation response map at each candidate to deterministically reject false periodic aliases.

## 9. Structural Verification
The engine computes Sobel edge gradients and runs a secondary structural similarity check on the top candidates to ensure the selection is structurally valid and not a high-contrast noise artifact.

## 10. Subpixel Refinement
A 2D parabolic curve is fitted to the local 3x3 ZNCC neighborhood around the winning candidate, allowing the engine to estimate the true peak position with continuous sub-pixel precision.

## 11. Runtime
Using an optimized OpenCV C++ `TM_CCOEFF_NORMED` implementation, the multi-scale grid search computes a megapixel search footprint in an average of ~3.8 seconds per sample on a standard CPU.

## 12. Evaluation Methodology
The evaluation uses a procedural generator to produce highly noisy periodic FinFET structures with random physical scale jitter (between 9.5x and 10.5x). Performance is strictly measured on these synthetic arrays.

## 13. 50-Sample Results
On the 50-sample synthetic evaluation, the algorithm achieved the following metrics:
- **Mean localization error**: 1.13 px
- **Median localization error**: 0.86 px
- **Maximum localization error**: 3.12 px
- **Accuracy <= 1 px**: 60.0%
- **Accuracy <= 2 px**: 78.0%
- **Accuracy <= 5 px**: 100.0%

## 14. Fresh Generalization Results
On a fresh random synthetic generalization set (100 samples), the algorithm consistently demonstrates a median error of < 1.0 px and > 98% accuracy within 5 pixels, confirming robust generalizability across random noise and scale seeds.

## 15. Limitations
- **Synthetic Data:** The evaluation uses procedurally generated synthetic data; results are not evidence of guaranteed production SEM performance. Never present synthetic results as real-wafer performance.
- **Scale Bounds:** The scale search is bounded between 9.5x and 10.5x.

## 16. Reproducibility
To perfectly reproduce the synthetic evaluation:
```bash
python generate_dataset.py --architecture finfet --num-pairs 50 --output-dir final_submission_50 --seed 2026
python run_submission_50.py
```

## 17. Installation
A Python 3.8+ runtime is required.
```bash
git clone https://github.com/Yukanthdayalan/drift-sense
cd drift-sense
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 18. Inference Usage
The evaluator-compatible command takes the reference and search images and outputs a single coordinate tuple to `stdout`:
```bash
python inference.py <reference.png> <search.png>
```
Example mandatory output:
```
(467.6238, 797.2276)
```
