import math
import unittest
import saa_eval as s


class ValidationTests(unittest.TestCase):
    def test_bad_tolerance(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                s.match_events([(0, 10)], [(5, 8)], tol_s=bad)

    def test_reversed_or_nonfinite_intervals(self):
        with self.assertRaises(ValueError):
            s.match_events([(10, 0)], [(5, 8)])
        with self.assertRaises(ValueError):
            s.match_events([(0, 10)], [(8, 5)])
        with self.assertRaises(ValueError):
            s.match_events([(0, math.nan)], [(5, 8)])

    def test_valid_inputs_unchanged(self):
        r = s.match_events([(0, 10), (20, 30)], [(2, 12), (21, 29)], tol_s=0.0)
        self.assertEqual(r["tp"], 2)
        self.assertEqual(s.match_events([(5, 5)], [(5, 5)], tol_s=0.0)["tp"], 1)  # zero length is allowed


if __name__ == "__main__":
    unittest.main()
