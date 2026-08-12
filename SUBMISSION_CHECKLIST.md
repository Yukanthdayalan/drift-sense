# Submission Checklist

This document maps all official Phase-2 presentation requirements to the files that satisfy them within this repository.

| Requirement | Status | Satisfying File / Details |
| :--- | :--- | :--- |
| **1. generate_dataset.py** |
| - Standalone script | **PASS** | `generate_dataset.py` (CLI wrapper) |
| - Accepts DRAM/FinFET architecture | **PASS** | `generate_dataset.py` (`--architecture finfet/dram`) |
| - Accepts number of pairs | **PASS** | `generate_dataset.py` (`--num-pairs`) |
| - Accepts output directory | **PASS** | `generate_dataset.py` (`--output-dir`) |
| - Records ground-truth center | **PASS** | Dataset generator correctly saves `ground_truth.json` |
| **2. inference.py** |
| - Standalone script | **PASS** | `inference.py` (Root directory) |
| - Accepts reference + search paths | **PASS** | `python inference.py <ref> <search>` |
| - Outputs ONLY (x,y) | **PASS** | Outputs `(x.xxxx, y.xxxx)` to stdout. Internal logs suppressed. |
| - No ground-truth dependency | **PASS** | Only reads image files from disk. Zero metadata leakage. |
| **3. requirements.txt** |
| - Clean installation | **PASS** | `requirements.txt` contains only production dependencies (`numpy`, `opencv-python`, `scipy`). |
| **4. README.md** |
| - Clone instructions | **PASS** | Documented in `README.md` |
| - Installation | **PASS** | Documented in `README.md` |
| - Dataset generation | **PASS** | Documented in `README.md` |
| - Inference execution | **PASS** | Documented in `README.md` |
| - Expected output | **PASS** | Documented in `README.md` |
| - Project explanation | **PASS** | Documented in `README.md` |
| **5. references/REFERENCES.md** |
| - Placeholder | **PASS** | Created `references/REFERENCES.md` |
| - No invented citations | **PASS** | Strict placeholder text used |
| **6. results/** |
| - final_50_results.csv | **PASS** | `results/final_50_results.csv` |
| - success_case.png | **PASS** | `results/success_case.png` |
| - failure_case.png | **PASS** | `results/failure_case.png` |
| **7. Repository Clean-Up** |
| - Remove development artifacts | **PASS** | Deleted legacy calibration scripts, old debug images, test sets, and profiling scripts. |

**Final Submission Audit Result: ALL CLEAR (PASS)**
