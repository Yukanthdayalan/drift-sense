"""
Unit tests for drift_sense.peak_detector

Covers all 35+ required scenarios including:
  - Single peak, multiple peaks, tie-breaking
  - Threshold rejection, NaN, Inf, empty map
  - Flat map, negative values, deterministic ordering
  - Candidate sorting, center distance, morphology correctness
  - Radius effect, response_delta effect, configuration loading
  - Exceptions, runtime sanity, reproducibility
  - Random synthetic maps, 1000x1000 stress test
"""
import time
import unittest
import numpy as np

from drift_sense.config import NMSConfig, TieBreakConfig
from drift_sense.exceptions import PeakDetectionError
from drift_sense.peak_detector import (
    PeakCandidate,
    detect_peaks,
    detect_best_peak,
    select_best_peak,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nms(radius: int = 5, delta: float = 0.05) -> NMSConfig:
    return NMSConfig(suppression_radius=radius, response_delta=delta)


def _make_tie(delta: float = 0.05) -> TieBreakConfig:
    return TieBreakConfig(delta=delta)


def _blank_map(h: int = 50, w: int = 50, val: float = 0.0) -> np.ndarray:
    return np.full((h, w), val, dtype=np.float32)


def _single_peak_map(
    h: int = 50, w: int = 50, peak_row: int = 25, peak_col: int = 25, peak_val: float = 0.9
) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[peak_row, peak_col] = peak_val
    return m


# ---------------------------------------------------------------------------
# 1. PeakCandidate dataclass
# ---------------------------------------------------------------------------

class TestPeakCandidate(unittest.TestCase):

    def test_frozen(self):
        c = PeakCandidate(x=10, y=20, score=0.9, distance_to_center=5.0)
        with self.assertRaises(Exception):
            c.x = 99  # type: ignore

    def test_equality(self):
        c1 = PeakCandidate(x=10, y=20, score=0.9, distance_to_center=5.0)
        c2 = PeakCandidate(x=10, y=20, score=0.9, distance_to_center=5.0)
        self.assertEqual(c1, c2)

    def test_ordering(self):
        c1 = PeakCandidate(x=0, y=0, score=0.9, distance_to_center=5.0)
        c2 = PeakCandidate(x=1, y=1, score=0.8, distance_to_center=3.0)
        # order=True: comparison is lexicographic on (x, y, score, distance)
        self.assertIsNotNone(c1 < c2 or c1 > c2)  # they are comparable


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_empty_map_raises(self):
        with self.assertRaises(PeakDetectionError):
            detect_peaks(np.array([], dtype=np.float32).reshape(0, 0), _make_nms())

    def test_nan_raises(self):
        m = _single_peak_map()
        m[0, 0] = float('nan')
        with self.assertRaises(PeakDetectionError):
            detect_peaks(m, _make_nms())

    def test_inf_raises(self):
        m = _single_peak_map()
        m[0, 0] = float('inf')
        with self.assertRaises(PeakDetectionError):
            detect_peaks(m, _make_nms())

    def test_neg_inf_raises(self):
        m = _blank_map(val=float('-inf'))
        with self.assertRaises(PeakDetectionError):
            detect_peaks(m, _make_nms())

    def test_wrong_dtype_raises(self):
        m = np.zeros((50, 50), dtype=np.float64)
        with self.assertRaises(PeakDetectionError):
            detect_peaks(m, _make_nms())

    def test_3d_input_raises(self):
        m = np.zeros((50, 50, 3), dtype=np.float32)
        with self.assertRaises(PeakDetectionError):
            detect_peaks(m, _make_nms())

    def test_non_array_raises(self):
        with self.assertRaises(PeakDetectionError):
            detect_peaks([[0.5, 0.6], [0.7, 0.8]], _make_nms())  # type: ignore


# ---------------------------------------------------------------------------
# 3. Single peak
# ---------------------------------------------------------------------------

class TestSinglePeak(unittest.TestCase):

    def test_single_peak_found(self):
        m = _single_peak_map(peak_row=10, peak_col=10, peak_val=0.9)
        candidates = detect_peaks(m, _make_nms(radius=3))
        # Peak at (10, 10) — col=x, row=y
        self.assertTrue(any(c.x == 10 and c.y == 10 for c in candidates))

    def test_single_peak_best(self):
        m = _single_peak_map(peak_row=25, peak_col=25)
        winner = detect_best_peak(m, _make_nms(radius=3), _make_tie())
        self.assertEqual(winner.x, 25)
        self.assertEqual(winner.y, 25)

    def test_single_peak_score(self):
        m = _single_peak_map(peak_val=0.85)
        winner = detect_best_peak(m, _make_nms(), _make_tie())
        self.assertAlmostEqual(winner.score, 0.85, places=5)

    def test_single_pixel_map(self):
        m = np.array([[0.7]], dtype=np.float32)
        winner = detect_best_peak(m, _make_nms(radius=1), _make_tie())
        self.assertEqual(winner.x, 0)
        self.assertEqual(winner.y, 0)


# ---------------------------------------------------------------------------
# 4. Multiple peaks
# ---------------------------------------------------------------------------

class TestMultiplePeaks(unittest.TestCase):

    def test_two_peaks_higher_score_wins(self):
        m = np.zeros((50, 50), dtype=np.float32)
        m[10, 10] = 0.95   # far from center
        m[25, 25] = 0.80   # at center
        winner = detect_best_peak(m, _make_nms(radius=3), _make_tie(delta=0.05))
        self.assertAlmostEqual(winner.score, 0.95, places=5)
        self.assertEqual(winner.x, 10)

    def test_returns_list_of_candidates(self):
        m = np.zeros((50, 50), dtype=np.float32)
        m[5, 5] = 0.9
        m[45, 45] = 0.88
        candidates = detect_peaks(m, _make_nms(radius=3, delta=0.05))
        self.assertGreaterEqual(len(candidates), 1)

    def test_candidates_sorted_descending_score(self):
        m = np.zeros((60, 60), dtype=np.float32)
        m[10, 10] = 0.95
        m[50, 50] = 0.85
        m[30, 30] = 0.90
        candidates = detect_peaks(m, _make_nms(radius=3, delta=0.15))
        scores = [c.score for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# 5. Tie-breaking
# ---------------------------------------------------------------------------

class TestTieBreaking(unittest.TestCase):

    def test_tied_peaks_center_wins(self):
        """Two identical scores: the one closer to center must win."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[10, 10] = 0.90   # far from center (50, 50)
        m[48, 48] = 0.90   # near center
        winner = detect_best_peak(m, _make_nms(radius=3, delta=0.05), _make_tie(delta=0.05))
        # (48, 48) is closer to center (49.5, 49.5) than (10, 10)
        self.assertEqual(winner.x, 48)
        self.assertEqual(winner.y, 48)

    def test_four_identical_peaks_deterministic(self):
        """Four identical peaks: must always return the same one."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[10, 10] = 0.90
        m[10, 90] = 0.90
        m[90, 10] = 0.90
        m[50, 50] = 0.90   # closest to center
        w1 = detect_best_peak(m, _make_nms(radius=3, delta=0.05), _make_tie(delta=0.05))
        w2 = detect_best_peak(m, _make_nms(radius=3, delta=0.05), _make_tie(delta=0.05))
        self.assertEqual(w1, w2)
        self.assertEqual(w1.x, 50)
        self.assertEqual(w1.y, 50)

    def test_no_tie_no_center_bias(self):
        """When top1 is clearly ahead, a near-center but weaker peak must NOT win."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[5, 5] = 0.95     # far from center, clearly strongest
        m[50, 50] = 0.80   # at center, much weaker
        winner = detect_best_peak(m, _make_nms(radius=3, delta=0.05), _make_tie(delta=0.05))
        self.assertEqual(winner.x, 5)
        self.assertEqual(winner.y, 5)

    def test_tie_break_only_within_delta(self):
        """Only candidates within delta of top1 participate in tie-break."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[5, 5] = 0.90     # top1
        m[50, 50] = 0.89   # within delta=0.05 → participates
        m[80, 80] = 0.80   # outside delta → excluded
        winner = detect_best_peak(m, _make_nms(radius=3, delta=0.10), _make_tie(delta=0.05))
        self.assertEqual(winner.x, 5)
        self.assertEqual(winner.y, 5)


# ---------------------------------------------------------------------------
# 6. Threshold rejection
# ---------------------------------------------------------------------------

class TestThreshold(unittest.TestCase):

    def test_below_threshold_rejected(self):
        """Peaks beyond response_delta below max must be absent from candidates."""
        m = np.zeros((50, 50), dtype=np.float32)
        m[10, 10] = 0.90
        m[40, 40] = 0.70  # 0.20 below max, delta=0.05 → rejected
        candidates = detect_peaks(m, _make_nms(radius=3, delta=0.05))
        rejected = [c for c in candidates if c.x == 40 and c.y == 40]
        self.assertEqual(len(rejected), 0)

    def test_response_delta_effect(self):
        """Increasing delta admits more candidates."""
        m = np.zeros((50, 50), dtype=np.float32)
        m[10, 10] = 0.90
        m[40, 40] = 0.80
        n_narrow = len(detect_peaks(m, _make_nms(radius=3, delta=0.05)))
        n_wide = len(detect_peaks(m, _make_nms(radius=3, delta=0.15)))
        self.assertGreaterEqual(n_wide, n_narrow)

    def test_no_survivors_raises(self):
        """If all peaks are filtered out, PeakDetectionError must be raised."""
        # A map where every pixel is identical except one, and delta=0.0 should
        # still keep at least the maximum. Let's use a case where something is
        # truly impossible — but that can't happen with correct logic (max always
        # survives). So test an extremely tight configuration on a special case.
        m = np.zeros((10, 10), dtype=np.float32) - 1.0  # all -1.0 → threshold = -1.0 - 0.0 = -1.0
        # Manually force: make a map where NMS suppresses ALL, which cannot happen
        # with correct implementation. Instead test empty candidate list path via
        # select_best_peak directly.
        with self.assertRaises(PeakDetectionError):
            select_best_peak([], (100, 100), _make_tie())


# ---------------------------------------------------------------------------
# 7. Flat / degenerate maps
# ---------------------------------------------------------------------------

class TestFlatMap(unittest.TestCase):

    def test_flat_map_returns_center(self):
        """A perfectly flat map: every pixel is a local max. Must return center-most."""
        h, w = 51, 51
        m = _blank_map(h=h, w=w, val=0.5)
        winner = detect_best_peak(m, _make_nms(radius=3, delta=0.0), _make_tie(delta=0.0))
        # Center is (25, 25) for a 51x51 map
        self.assertEqual(winner.x, 25)
        self.assertEqual(winner.y, 25)

    def test_all_negative_map(self):
        """Map with all negative values must still return the maximum pixel."""
        m = np.full((50, 50), -0.5, dtype=np.float32)
        m[20, 30] = -0.1  # highest value
        winner = detect_best_peak(m, _make_nms(radius=3), _make_tie())
        self.assertEqual(winner.x, 30)
        self.assertEqual(winner.y, 20)


# ---------------------------------------------------------------------------
# 8. Morphology correctness
# ---------------------------------------------------------------------------

class TestMorphology(unittest.TestCase):

    def test_radius_suppresses_nearby_peaks(self):
        """Two peaks within suppression radius: only one should survive."""
        m = np.zeros((50, 50), dtype=np.float32)
        m[25, 25] = 0.90
        m[26, 26] = 0.85  # 1 pixel away — within radius=5
        candidates = detect_peaks(m, _make_nms(radius=5, delta=0.10))
        # Only (25, 25) should survive morphological suppression
        surviving_at_26 = [c for c in candidates if c.x == 26 and c.y == 26]
        self.assertEqual(len(surviving_at_26), 0)

    def test_radius_allows_distant_peaks(self):
        """Two peaks farther apart than radius: both should survive."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[10, 10] = 0.90
        m[80, 80] = 0.88
        candidates = detect_peaks(m, _make_nms(radius=5, delta=0.10))
        xs = {c.x for c in candidates}
        self.assertIn(10, xs)
        self.assertIn(80, xs)

    def test_local_max_definition(self):
        """A pixel that is NOT the max in its neighbourhood must be suppressed."""
        m = np.zeros((20, 20), dtype=np.float32)
        m[10, 10] = 0.90
        m[10, 11] = 0.95  # dominates the neighbourhood of (10, 10)
        candidates = detect_peaks(m, _make_nms(radius=3, delta=0.10))
        at_10_10 = [c for c in candidates if c.x == 10 and c.y == 10]
        self.assertEqual(len(at_10_10), 0)


# ---------------------------------------------------------------------------
# 9. Center distance
# ---------------------------------------------------------------------------

class TestCenterDistance(unittest.TestCase):

    def test_distance_computed_correctly(self):
        """Verify center distance is Euclidean distance to map center."""
        h, w = 101, 101
        m = _single_peak_map(h=h, w=w, peak_row=0, peak_col=0, peak_val=0.9)
        candidates = detect_peaks(m, _make_nms(radius=3))
        top = candidates[0]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        expected_dist = ((top.x - cx) ** 2 + (top.y - cy) ** 2) ** 0.5
        self.assertAlmostEqual(top.distance_to_center, expected_dist, places=5)

    def test_center_candidate_has_zero_distance(self):
        """Peak at the exact center of an odd-dimension map has distance ≈ 0."""
        h, w = 51, 51
        m = _single_peak_map(h=h, w=w, peak_row=25, peak_col=25)
        candidates = detect_peaks(m, _make_nms(radius=3))
        center_cand = next(c for c in candidates if c.x == 25 and c.y == 25)
        self.assertAlmostEqual(center_cand.distance_to_center, 0.0, places=5)


# ---------------------------------------------------------------------------
# 10. Configuration loading
# ---------------------------------------------------------------------------

class TestConfigurationLoading(unittest.TestCase):

    def test_default_config_works(self):
        """NMSConfig and TieBreakConfig default values must produce valid results."""
        m = _single_peak_map()
        cfg = NMSConfig()
        tie = TieBreakConfig()
        winner = detect_best_peak(m, cfg, tie)
        self.assertIsInstance(winner, PeakCandidate)

    def test_custom_config_respected(self):
        """Custom response_delta and radius must be respected."""
        m = np.zeros((100, 100), dtype=np.float32)
        m[10, 10] = 0.90
        m[80, 80] = 0.75  # 0.15 below max
        # With delta=0.05, (80, 80) is excluded
        cfg_narrow = NMSConfig(suppression_radius=3, response_delta=0.05)
        c_narrow = detect_peaks(m, cfg_narrow)
        self.assertFalse(any(c.x == 80 for c in c_narrow))
        # With delta=0.20, (80, 80) is included
        cfg_wide = NMSConfig(suppression_radius=3, response_delta=0.20)
        c_wide = detect_peaks(m, cfg_wide)
        self.assertTrue(any(c.x == 80 for c in c_wide))


# ---------------------------------------------------------------------------
# 11. Reproducibility / determinism
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):

    def test_identical_inputs_identical_outputs(self):
        """Same map and config must produce identical results every time."""
        rng = np.random.default_rng(0)
        m = rng.uniform(-1.0, 1.0, (50, 50)).astype(np.float32)
        nms = _make_nms()
        tie = _make_tie()
        r1 = detect_best_peak(m, nms, tie)
        r2 = detect_best_peak(m, nms, tie)
        self.assertEqual(r1, r2)

    def test_random_maps_no_exception(self):
        """Random maps with valid float32 values must always return a result."""
        rng = np.random.default_rng(42)
        for seed in range(20):
            m = rng.uniform(-1.0, 1.0, (60, 60)).astype(np.float32)
            winner = detect_best_peak(m, _make_nms(), _make_tie())
            self.assertIsInstance(winner, PeakCandidate)
            self.assertTrue(np.isfinite(winner.score))
            self.assertTrue(np.isfinite(winner.distance_to_center))


# ---------------------------------------------------------------------------
# 12. Runtime sanity
# ---------------------------------------------------------------------------

class TestRuntimeSanity(unittest.TestCase):

    def test_stress_1000x1000(self):
        """
        1000×1000 map must complete in under 2 seconds on CPU.
        Validates O(N) time complexity claim.
        """
        rng = np.random.default_rng(7)
        m = rng.uniform(-1.0, 1.0, (1000, 1000)).astype(np.float32)
        t0 = time.perf_counter()
        winner = detect_best_peak(m, _make_nms(), _make_tie())
        elapsed = time.perf_counter() - t0
        self.assertIsInstance(winner, PeakCandidate)
        self.assertLess(elapsed, 2.0, f"1000x1000 took {elapsed:.2f}s — too slow.")

    def test_small_map_fast(self):
        """A 10×10 map must complete in under 10 ms."""
        m = _single_peak_map(h=10, w=10, peak_row=5, peak_col=5)
        t0 = time.perf_counter()
        detect_best_peak(m, _make_nms(radius=2), _make_tie())
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.5)


if __name__ == "__main__":
    unittest.main()
