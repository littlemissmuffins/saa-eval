"""Regression tests for the bounded saa-eval patch (synthetic inputs only; no real-data scores)."""
import itertools
import math
import os
import subprocess
import sys
import tempfile
import unittest

from saa_eval import (blocked_folds, intervals_to_mask, mask_to_intervals, match_events,
                      sample_level_scores)

HERE = os.path.dirname(os.path.abspath(__file__))


def candidate(r, p, tol_s):
    """Independent restatement of the historical candidate rule (overlap with widened ref)."""
    a, b = r[0] - tol_s, r[1] + tol_s
    return min(b, p[1]) - max(a, p[0]) > 0 or a <= p[0] <= b or p[0] <= a <= p[1]


def brute_max(ref, pred, tol_s):
    """Max matches over every partial injective assignment ref -> pred-or-None."""
    best = 0
    for assign in itertools.product([None] + list(range(len(pred))), repeat=len(ref)):
        used = [j for j in assign if j is not None]
        if len(used) != len(set(used)):
            continue
        if all(candidate(ref[i], pred[j], tol_s) for i, j in enumerate(assign) if j is not None):
            best = max(best, len(used))
    return best


class Matching(unittest.TestCase):
    REF = [(0, 1000), (1100, 1200)]
    PRED = [(500, 1150), (0, 400)]

    def test_greedy_default_unchanged(self):
        for kw in ({}, {"matching": "greedy"}):
            g = match_events(self.REF, self.PRED, 60, **kw)
            self.assertEqual((g["tp"], g["fp"], g["fn"]), (1, 1, 1))
            self.assertEqual([m[:2] for m in g["matches"]], [(0, 0)])

    def test_maximum_finds_two(self):
        m = match_events(self.REF, self.PRED, 60, matching="maximum")
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 0, 0))
        self.assertEqual([x[:2] for x in m["matches"]], [(0, 1), (1, 0)])
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_invalid_matching_mode(self):
        with self.assertRaises(ValueError):
            match_events(self.REF, self.PRED, matching="hungarian")

    def test_boundary_overlap_semantics_unchanged(self):
        ref = [(0, 900)]
        expect = {(600, 700): True, (-3000, 30): True, (950, 5000): True,
                  (960, 1000): True, (961, 1000): False, (-61, -60): True, (-70, -61): False}
        for mode in ("greedy", "maximum"):
            for p, ok in expect.items():
                self.assertEqual(bool(match_events(ref, [p], 60, matching=mode)["tp"]), ok, (mode, p))
        m = match_events(ref, [(600, 700)], 60, matching="maximum")["matches"][0]
        self.assertAlmostEqual(m[2], 10.0)
        self.assertAlmostEqual(m[3], -10 / 3)

    def test_exhaustive_family_vs_bruteforce(self):
        pool = [(0, 1000), (1100, 1200), (500, 1150), (0, 400), (1260, 1260), (1300, 1500)]
        subsets = [c for k in range(4) for c in itertools.combinations(pool, k)]  # 42 subsets
        cases = greedy_short = unequal = 0
        for ref in subsets:
            for pred in subsets:
                if len(ref) + len(pred) > 4:  # every pair with at most 4 events total: 764 cases
                    continue
                cases += 1
                unequal += len(ref) != len(pred)
                ref_l, pred_l = list(ref), list(pred)
                m1 = match_events(ref_l, pred_l, 60, matching="maximum")
                m2 = match_events(ref_l, pred_l, 60, matching="maximum")
                g = match_events(ref_l, pred_l, 60)
                self.assertEqual(m1["matches"], m2["matches"])  # deterministic
                self.assertEqual(m1["tp"], brute_max(ref_l, pred_l, 60), (ref, pred))
                pairs = [x[:2] for x in m1["matches"]]
                self.assertEqual(len({i for i, _ in pairs}), len(pairs))
                self.assertEqual(len({j for _, j in pairs}), len(pairs))
                self.assertTrue(all(candidate(ref_l[i], pred_l[j], 60) for i, j in pairs))
                self.assertEqual(m1["fp"] + m1["tp"], len(pred_l))
                self.assertEqual(m1["fn"] + m1["tp"], len(ref_l))
                self.assertLessEqual(g["tp"], m1["tp"])
                greedy_short += g["tp"] < m1["tp"]
        print(f"\n  exhaustive family: {cases} cases, {unequal} with unequal ref/pred counts, "
              f"greedy short of maximum in {greedy_short}", file=sys.stderr)
        self.assertLessEqual(cases, 1000)
        self.assertGreater(unequal, 0)
        self.assertGreater(greedy_short, 0)


class MasksAndGaps(unittest.TestCase):
    def test_unequal_masks_rejected(self):
        with self.assertRaises(ValueError):
            sample_level_scores([True] * 10, [True] * 5)
        with self.assertRaises(ValueError):
            sample_level_scores([True] * 10, [True] * 10, covered=[True] * 3)
        with self.assertRaises(ValueError):
            sample_level_scores([True] * 10, [True] * 10, covered=[True] * 20)
        with self.assertRaises(ValueError):
            mask_to_intervals([0, 1, 2], [True, True])

    def test_nonfinite_and_unsorted_times_rejected(self):
        for bad in ([5, 1, 2], [0, float("nan"), 2], [0, float("inf")], [0, -math.inf]):
            with self.assertRaises(ValueError, msg=bad):
                intervals_to_mask(bad, [(0, 3)])
            with self.assertRaises(ValueError, msg=bad):
                mask_to_intervals(bad, [True] * len(bad))
            with self.assertRaises(ValueError, msg=bad):
                blocked_folds(bad, 1, 0)
        self.assertEqual(intervals_to_mask([0, 1, 1, 2], [(1, 1)]), [False, True, True, False])

    def test_missing_observation_gap(self):
        times, mask = [0, 10, 20, 5000, 5010], [True] * 5
        self.assertEqual(mask_to_intervals(times, mask), [(0, 5010)])  # default unchanged
        self.assertEqual(mask_to_intervals(times, mask, max_gap_s=30), [(0, 20), (5000, 5010)])
        self.assertEqual(mask_to_intervals(times, mask, max_gap_s=10), [(0, 20), (5000, 5010)])
        self.assertEqual(mask_to_intervals(times, mask, max_gap_s=9.9),
                         [(0, 0), (10, 10), (20, 20), (5000, 5000), (5010, 5010)])
        # a False sample between positives still splits regardless of gap limit
        self.assertEqual(mask_to_intervals([0, 10, 20], [True, False, True], max_gap_s=1e9),
                         [(0, 0), (20, 20)])
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mask_to_intervals(times, mask, max_gap_s=bad)


class Folds(unittest.TestCase):
    TIMES = list(range(0, 4000, 10))
    PASS = (1500, 2400)

    def _leaks(self, folds, ivs):
        out = []
        for train, test in folds:
            ttimes = [self.TIMES[i] for i in test]
            for s, e in ivs:
                if any(s <= t <= e for t in ttimes) and any(s <= self.TIMES[i] <= e for i in train):
                    out.append((s, e))
        return out

    def test_time_buffer_only_still_leaks_long_pass(self):
        folds = blocked_folds(self.TIMES, 4, 60)
        self.assertTrue(self._leaks(folds, [self.PASS]))
        self.assertEqual(folds, blocked_folds(self.TIMES, 4, 60, event_intervals=None))

    def test_event_purged_no_leak(self):
        events = [self.PASS, (300, 400), (3990, 3990)]
        folds = blocked_folds(self.TIMES, 4, 60, event_intervals=events)
        self.assertEqual(self._leaks(folds, events), [])
        for k, (train, test) in enumerate(folds):
            self.assertFalse(set(train) & set(test))
            t_lo, t_hi = self.TIMES[test[0]], self.TIMES[test[-1]]
            for i in train:
                t = self.TIMES[i]
                self.assertFalse(t_lo - 60 <= t <= t_hi + 60)
                for s, e in events:
                    if s <= t_hi and e >= t_lo:
                        self.assertFalse(s - 60 <= t <= e + 60, (k, t, (s, e)))
        # fold 1 (1000-1990) touches the pass, so training must lose 2060..2460 too
        train1 = {self.TIMES[i] for i in folds[1][0]}
        self.assertNotIn(2400, train1)
        self.assertNotIn(2460, train1)
        self.assertIn(2470, train1)
        # an event not touching the test block does not purge anything extra
        f0_plain = blocked_folds(self.TIMES, 4, 60)[0]
        self.assertEqual(blocked_folds(self.TIMES, 4, 60, event_intervals=[(3000, 3100)])[0], f0_plain)

    def test_invalid_fold_requests(self):
        for n_folds in (0, -1, 3.0, True, 4):
            with self.assertRaises(ValueError, msg=n_folds):
                blocked_folds([0, 10, 20], n_folds, 0)
        for buf in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=buf):
                blocked_folds([0, 10, 20], 1, buf)
        with self.assertRaises(ValueError):
            blocked_folds([], 1, 0)
        for ev in ([(10, 5)], [(0, float("nan"))]):
            with self.assertRaises(ValueError, msg=ev):
                blocked_folds([0, 10, 20], 1, 0, event_intervals=ev)
        self.assertEqual(len(blocked_folds([0, 10, 20], 3, 0)), 3)


class CLI(unittest.TestCase):
    def test_alias_and_whitespace(self):
        csvtext = ("annotator,start_utc,end_utc\n"
                   "Martin,2020-10-15T00:00:00Z,2020-10-15T00:10:00Z\n"
                   "Hitaeshi Sehgal,2020-10-15T00:00:30Z,2020-10-15T00:10:30Z\n")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(csvtext)
        try:
            for ref, pred in (("martin kakona", "hitaeshi sehgal"), ("Martin", "hitaeshi sehgal"),
                              (" MARTIN  Kakona ", "Hitaeshi  Sehgal")):
                p = subprocess.run([sys.executable, os.path.join(HERE, "compare_marks.py"), f.name,
                                    "--ref", ref, "--pred", pred], capture_output=True, text=True, cwd=HERE)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("matched: 1", p.stdout)
            p = subprocess.run([sys.executable, os.path.join(HERE, "compare_marks.py"), f.name,
                                "--ref", "nobody", "--pred", "martin"], capture_output=True, text=True, cwd=HERE)
            self.assertEqual(p.returncode, 2)
            self.assertIn("not found", p.stderr)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()

