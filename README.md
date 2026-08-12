# Drift-Sense: Precision Wafer Localization Engine

Drift-Sense is an advanced, scale-invariant image registration engine designed for Applied Materials. It is engineered to localize extremely periodic and highly distorted sub-pitch semiconductor structures (FinFETs/DRAM) inside large search areas, without relying on external metadata.

The system utilizes an innovative **Top-2 Sharpness Verification** strategy to resolve structural sub-pitch aliasing by dynamically analyzing local peak sharpness and center-aware disambiguation.

## Phase-2 Final Evaluator Submission

This repository contains the standalone, fully isolated inference module and synthetic dataset generator for Phase-2 evaluation.

### 1. Installation

A clean installation is recommended to guarantee environment isolation. Only production dependencies are required (`numpy`, `opencv-python`, `scipy`).

```bash
# Clone the repository
git clone <repository_url>
cd drift_sense

# Create a clean virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Generation

You can procedurally generate completely unobserved synthetic periodic defect datasets (FinFET / DRAM) for blind evaluation.

```bash
python generate_dataset.py --architecture finfet --num-pairs 30 --output-dir evaluation_dataset
```
The script will create pairs of `reference.png` and `search.png`. The absolute ground-truth coordinate is recorded in each sample's `ground_truth.json` (note: the inference engine is strictly forbidden from accessing this metadata).

### 3. Inference Execution

The inference engine runs completely standalone. It parses the reference and search images directly from disk and computes the optimized match entirely offline.

```bash
python inference.py <path_to_reference.png> <path_to_search.png>
```

#### Expected Output
The script suppresses all internal verbose logging to maintain a clean machine-readable pipe. It will output exactly one line to `stdout`:
```
(x.xxxx, y.xxxx)
```
*Example: `(580.0909, 602.5225)`*

This coordinate represents the exact predicted center of the reference region mapped onto the search coordinate plane.

### 4. Evaluation Context

In the final 50-sample independent evaluation, Drift-Sense achieved:
- **100.0% Success Rate** ($\le 5$ pixel tolerance limit)
- **Max Absolute Error**: ~3.076 px
- **Average Runtime**: ~2.7s per sample on CPU

Please refer to `results/FINAL_50_SAMPLE_REPORT.md` (if available) for the deep-dive failure/success analysis.
