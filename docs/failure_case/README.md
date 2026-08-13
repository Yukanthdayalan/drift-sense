# Failure Case: sample_021 (True Leakage-Free Eval)

This sample demonstrates a genuine failure mode under 1.75x noise stress. The multi-scale ZNCC backbone generated a candidate so heavily distorted by Speckle and Gaussian noise that the topological sharpness threshold tied. Because the training dataset was properly isolated (no train/eval leakage), the CNN disambiguator lacked the exact noise-memorization advantage and was forced to generalize. It ultimately selected a false periodic peak with a confidence score of 0.4476. Crucially, the low_confidence flag correctly triggered, alerting downstream tools to the uncertainty.

- **Ground Truth:** (680.0000, 356.0000)
- **Predicted:** (680.0994, 357.9678)
- **Error:** 1.9703 pixels
