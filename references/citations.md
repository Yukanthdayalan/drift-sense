# Citations & Academic References for Drift-Sense Pipeline

The synthetic generation pipeline and algorithmic choices in Drift-Sense are grounded in established physics and literature regarding scanning electron microscopy and template matching.

### 1. Speckle / Electron Shot Noise Modeling
*Note: True "speckle" is a coherent imaging phenomenon (e.g. SAR/Ultrasound). In SEM, granular noise is more accurately modeled as Poisson shot noise interacting with detector Gaussian noise, which manifests similarly to multiplicative speckle.*
**Citation:** Goldstein, J. I., et al. (2017). *Scanning Electron Microscopy and X-Ray Microanalysis* (4th ed.). Springer.
**Justification:** Justifies the use of mixed noise distribution assumptions and intensity-dependent variance representing the finite electron counting statistics (`dataset.py:L142` - Gaussian/Speckle-like additive noise combination).

### 2. General Gaussian/Sensor Noise in SEM
**Citation:** Pizarro, L., et al. (2010). "A highly robust noise model for scanning electron microscopy image restoration." *Medical Image Analysis*, 14(5), 633-644.
**Justification:** Justifies the baseline addition of Gaussian noise to simulate the electronic thermal noise found in standard SEM photomultiplier tubes and solid-state detectors, particularly scaling differently for the search and reference domains based on dwell times (`dataset.py:L146` - `_add_noise` parameters).

### 3. Edge Brightening (Edge Effect)
**Citation:** Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis* (2nd ed.). Springer Series in Optical Sciences.
**Justification:** Documents the physical phenomenon where topographic edges yield higher secondary electron (SE) escape probability. This strictly justifies the edge-brightening synthesis logic using Sobel gradients to boost intensity along discontinuities (`dataset.py:L82-L95`).

### 4. FinFET / DRAM Structural Periodicity
**Citation:** Orji, N. G., et al. (2018). "Metrology for the next generation of semiconductor devices." *Nature Electronics*, 1(9), 532-547.
**Justification:** Discusses the repetitive 3D FinFET array layout (sub-20nm pitch structures) and the specific CD-SEM metrology challenges associated with dense, high-aspect-ratio periodic geometries, justifying the grid layout generation parameters in the synthetic data (`dataset.py:L34-L52`).

### 5. Template Matching Ambiguity on Periodic Structures
**Citation:** Brunelli, R. (2009). *Template Matching Techniques in Computer Vision: Theory and Practice*. John Wiley & Sons.
**Justification:** Discusses the fundamental vulnerability of normalized cross-correlation (NCC) to auto-correlation ambiguity when applied to strictly repeating periodic structures. This justifies the necessity of the hybrid CNN disambiguator to handle near-tied ZNCC peaks (`matcher.py:L102`).

### 6. Wafer Navigation and Die-to-Database Inspection
**Citation:** Villarrubia, J. S., & Vladar, A. E. (2010). "The dependence of scanning electron microscope imaging on sample topology." *Scanning*, 32(3), 141-150.
**Justification:** Provides context on SEM navigation errors and structural localization, validating the hackathon's core problem statement of detecting reference offsets from drift/stage-error in high-noise periodic fields.
