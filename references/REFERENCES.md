# References

[1] J. P. Lewis, "Fast Normalized Cross-Correlation," *Vision Interface*, vol. 10, no. 1, pp. 120–123, 1995.
Relevance: This foundational paper provides the theoretical basis for computing Normalized Cross-Correlation (ZNCC) efficiently in the frequency domain using the Fast Fourier Transform (FFT). Drift-Sense implements this exactly in `zncc.py` to achieve rapid localization of templates within megapixel search images.

[2] R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. Springer International Publishing, 2022, doi: 10.1007/978-3-030-34372-9.
Relevance: This textbook is the authoritative source for the standard computer vision processing pipeline used throughout the repository. It specifically supports the Z-score normalization strategy implemented in `preprocess.py` and the `INTER_AREA` downsampling methodologies used for image pyramids in `geometry.py`.

[3] B. Xiong, Q. Zhang, and V. Baltazart, "On Quadratic Interpolation of Image Cross-Correlation for Subpixel Motion Extraction," *Sensors*, vol. 22, no. 3, p. 1274, 2022, doi: 10.3390/s22031274.
Relevance: This reference directly analyzes the mathematical validity of 2D quadratic surface fitting on cross-correlation peaks. It supports the algorithm implemented in `subpixel.py` which extracts sub-pixel precision from integer-level discrete correlation peaks.

[4] Q. Tian and M. N. Huhns, "Algorithms for subpixel registration," *Computer Vision, Graphics, and Image Processing*, vol. 35, no. 2, pp. 220–233, 1986, doi: 10.1016/0734-189X(86)90028-9.
Relevance: A classic paper detailing the numerical error bounds of intensity and correlation interpolation techniques. It supports the deterministic subpixel fallback algorithms used in `subpixel.py` when primary peak curvature is ill-conditioned.

[5] A. Rosenfeld and G. J. Vanderbrug, "Coarse-fine template matching," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 7, no. 2, pp. 104–107, 1977, doi: 10.1109/TSMC.1977.4309665.
Relevance: Provides the theoretical framework for hierarchical search optimizations. This directly supports the coarse-to-fine scale bracket search optimization implemented in `scale_search.py` to prevent redundant full-resolution correlations.

[6] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui, "Visual object tracking using adaptive correlation filters," in *2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2544–2550, 2010, doi: 10.1109/CVPR.2010.5539960.
Relevance: This paper introduced the Peak-to-Sidelobe Ratio (PSR) for correlation maps. Drift-Sense utilizes this exact mathematical metric in `peak_detector.py` to measure peak sharpness and deterministically break ties between aliased structural matches.

[7] A. Neubeck and L. Van Gool, "Efficient Non-Maximum Suppression," in *18th International Conference on Pattern Recognition (ICPR'06)*, vol. 3, pp. 850–855, 2006, doi: 10.1109/ICPR.2006.479.
Relevance: Analyzes optimal strategies for Non-Maximum Suppression (NMS) on 2D grids. This supports the localized peak-finding and masking algorithms implemented in `peak_detector.py` to isolate distinct candidate matches on the ZNCC response map.

[8] P. Welch, "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms," *IEEE Transactions on Audio and Electroacoustics*, vol. 15, no. 2, pp. 70–73, 1967, doi: 10.1109/TAU.1967.1161901.
Relevance: The foundational reference for Power Spectral Density (PSD) analysis via FFT. This supports the `_compute_psd_scale_prior` function in `scale_search.py`, which dynamically estimates the periodic pitch of FinFET structures to seed the scale search.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004, doi: 10.1109/TIP.2003.819861.
Relevance: Formalizes the Structural Similarity (SSIM) index and the use of gradient map comparisons. This informs the structural verification techniques implemented in `verification.py` to independently validate matches post-localization.

[10] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, vol. 25, no. 11, pp. 120–125, 2000.
Relevance: The foundational citation for the core open-source library executing the low-level accelerated primitives (e.g., `cv2.matchTemplate`, `cv2.resize`) upon which the pure-Python orchestrator is built.

[11] J. Li et al., "Review of wafer defect detection in semiconductor manufacturing: Algorithms, systems, and data," *Journal of Intelligent Manufacturing*, 2026, doi: 10.1007/s10845-026-02845-z.
Relevance: Provides authoritative context on the industrial application of template matching and computer vision for wafer inspection. This validates the overarching problem domain, target constraints, and defect-aware matching strategies implemented in `dataset.py` and `matcher.py`.

## Reference-to-Implementation Mapping

| Reference | Drift-Sense Component | Supported Concept |
|---|---|---|
| [1] | `zncc.py` | ZNCC / FFT-based normalized correlation |
| [2] | `preprocess.py`, `geometry.py` | Z-score normalization / Image Pyramids |
| [3] | `subpixel.py` | 2D quadratic interpolation for peak refinement |
| [4] | `subpixel.py` | Error bounds for subpixel registration |
| [5] | `scale_search.py` | Coarse-to-fine multi-scale search |
| [6] | `peak_detector.py` | Peak-to-Sidelobe Ratio (PSR) for tie-breaking |
| [7] | `peak_detector.py` | Efficient Non-Maximum Suppression (NMS) |
| [8] | `scale_search.py` | PSD / FFT periodicity estimation |
| [9] | `verification.py` | Structural similarity and gradient verification |
| [10]| Core CV logic | Fast C++ primitives (OpenCV) |
| [11]| `dataset.py`, `matcher.py`| Semiconductor wafer inspection domain context |
