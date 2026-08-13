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
Drift-Sense uses a hybrid architecture: a fast, deterministic classical CV pipeline (Z-score normalization + multi-scale FFT-ZNCC grid search) handles the bulk of the localization, combined with a lightweight CNN-based periodic-ambiguity resolver that activates exclusively on ambiguous cases.

## 7. Candidate Generation
Instead of assuming the highest correlation peak is correct, Drift-Sense uses spatial Non-Maximum Suppression (NMS) to generate multiple plausible candidates across the search grid.

## 8. Ambiguity Resolution
Candidates within a tight intensity threshold of the absolute maximum ZNCC score trigger the ambiguity resolver. For these challenging periodic cases, a lightweight PyTorch CNN processes small 64x64 crops of the candidates from the search image to deterministically break the tie, falling back to a classical topological `sharpness` heuristic if the model is absent or uncertain.

## 9. Structural Verification
The engine computes Sobel edge gradients and runs a secondary structural similarity check on the top candidates to ensure the selection is structurally valid and not a high-contrast noise artifact.

## 10. Subpixel Refinement
A 2D parabolic curve is fitted to the local 3x3 ZNCC neighborhood around the winning candidate, allowing the engine to estimate the true peak position with continuous sub-pixel precision.

## 11. Runtime
Using an optimized OpenCV C++ `TM_CCOEFF_NORMED` implementation, the multi-scale grid search computes a megapixel search footprint in an average of ~3.8 seconds per sample on a standard CPU.

## 12. Evaluation Methodology
The evaluation uses a procedural generator to produce highly noisy periodic FinFET structures with random physical scale jitter (between 9.5x and 10.5x). Performance is strictly measured on these synthetic arrays.

## 13. Hybrid Pipeline & Stress-Test Results
On synthetic evaluation datasets, the approach provides remarkable stability, even under simulated hardware stress (1.75x noise injection):

### Baseline (1x Noise, 30 pairs)
- **Mean localization error**: 0.42 px
- **Median localization error**: 0.40 px
- **Maximum localization error**: 0.62 px
- **Accuracy <= 1 px**: 100.0%

### Stress Test (1.75x Noise, 30 pairs)
- **Mean localization error**: 0.49 px
- **Median localization error**: 0.59 px
- **Maximum localization error**: 0.59 px
- **Accuracy <= 1 px**: 100.0%

### Ambiguous Subset Analysis
We implemented a hybrid CNN disambiguator designed to resolve periodic-ambiguity ties that the deterministic ZNCC backbone cannot break with correlation score alone. Rigorous testing (after fixing a scale-search artifact that had inflated the apparent ambiguity rate) showed genuine periodic ties are infrequent in this dataset's noise regime (approximately 4/30 samples), and are already resolved correctly by a lightweight sharpness heuristic at this density. The CNN architecture and training pipeline are implemented and validated for correctness, but not yet shown to outperform the heuristic at the current ambiguity frequency — we view it as a scalable extension for denser, more periodic real-fab layouts where tie frequency is expected to increase, rather than a currently load-bearing component. The classical backbone alone achieves 100.0% accuracy at <=2px on the 1.75x noise stress test.

## 14. Fresh Generalization Results
On a fresh random synthetic generalization set (100 samples, different seed from the training/evaluation datasets), the algorithm consistently demonstrates robust generalizability across random noise and scale seeds. The results are logged in `results/generalization_results.csv`.

| Metric | Value |
|--------|-------|
| **Mean Error** | 0.03 px |
| **Median Error** | 0.02 px |
| **Max Error** | 0.14 px |
| **Accuracy <= 1 px** | 100.0% |
| **Accuracy <= 2 px** | 100.0% |
| **Accuracy <= 5 px** | 100.0% |

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
Verified working on Python 3.11.9 (recommended). Python 3.14+ is known NOT to work due to missing prebuilt wheels for pinned dependencies as of this writing. Other versions (3.9–3.13) have not been explicitly tested — if using a version other than 3.11, verify `pip install -r requirements.txt` completes cleanly before relying on it.
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

## 19. Disambiguator Training
To train the CNN disambiguator on a newly generated dataset:
```bash
python generate_dataset.py --num-pairs 100 --output-dir dataset_train
python train_disambiguator.py
```
The model will be saved to `models/disambiguator.pt` and automatically picked up by `inference.py` (graceful fallback to classical sharpness if missing). Modules:
- `train_disambiguator.py`: Generates positive and hard-negative crops for training the CNN.
- `src/drift_sense/disambiguator.py`: Defines the PyTorch CNN architecture and inference wrapper.
