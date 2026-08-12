# Drift-Sense Pipeline Architecture

The Drift-Sense localization engine operates strictly on classical computer vision principles without relying on machine learning. It uses a sequence of deterministic processing stages to securely narrow down, disambiguate, and refine the location of a target reference patch within a search image.

## Data Flow

```
Reference Image                 Search Image
      │                              │
      └───────────┐      ┌───────────┘
                  ▼      ▼
                Validation
                  │
                  ▼
              Preprocessing
(Z-Score Normalization, optional filtering)
                  │
                  ▼
             Scale Search
(Coarse grid → Fine grid over scaling bounds)
                  │
                  ▼
          Template Scaling
(Reference scaled to Matcher resolution)
                  │
                  ▼
              FFT-ZNCC
(Fast Normalized Cross-Correlation in Frequency Domain)
                  │
                  ▼
             Response Map
(Score array bounded between -1.0 and 1.0)
                  │
                  ▼
         NMS Peak Detection
(Morphological Dilation Non-Maximum Suppression)
                  │
                  ▼
          Candidate Ranking
                  │
                  ▼
        Periodic Tie-Breaking
(Deterministic proximity resolution for tied peaks)
                  │
                  ▼
             Integer Peak
(Top-Left integer coordinate of matched patch)
                  │
                  ▼
        Sub-Pixel Refinement
(1D Parabolic Interpolation for Sub-Pixel Accuracy)
                  │
                  ▼
        Structural Verification
(Sobel-Gradient Structural Similarity Check)
                  │
                  ▼
          Center Coordinate
(Geometric shift from Top-Left to Target Center)
                  │
                  ▼
         Final Output (x, y)
```

## Description of Stages

1. **Validation**: Validates array inputs, bounds, dimensions, and type constraints to prevent silent failures.
2. **Preprocessing**: Normalizes arrays (0 mean, 1 std) via zero-score standardization to ensure intensity shifts do not skew correlation.
3. **Scale Search**: Generates a coarse scale hypothesis using a lower-resolution proxy sweep, and refines it using a denser local search array bounded between 9.0x and 11.0x.
4. **Template Scaling**: Down-samples or up-samples the pristine reference image directly based on the final scale multiplier.
5. **FFT-ZNCC**: Performs normalized cross-correlation in the frequency domain. `scipy.signal.fftconvolve(..., mode='valid')` is strictly used to assure bounding overlap arithmetic is perfectly aligned.
6. **NMS Peak Detection**: Exposes secondary alignment candidates using cv2.dilate instead of collapsing prematurely to the global maximum, essential for periodic structures.
7. **Tie-Breaking**: Identifies statistically tied correlation structures (threshold default 0.05) and enforces physical determinism by evaluating relative Euclidean proximity to the central origin of the image frame.
8. **Sub-Pixel Refinement**: Solves parabolic vertex arrays across X and Y dimensions independently to retrieve non-integer fractions.
9. **Structural Verification**: Evaluates the localized candidate patch against the scaled template utilizing structural (gradient) gradients, ensuring high-frequency feature fidelity independent of simple luminescence.
10. **Center Output**: The engine implicitly identifies the top-left index. The final stage maps this to the topological center per pipeline requirements.
