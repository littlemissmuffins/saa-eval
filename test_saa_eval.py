import os, tempfile, unittest
from saa_eval import *

class T(unittest.TestCase):
    def test_parse_and_names(self):
        self.assertEqual(parse_ts("1970-01-01T00:01:00.500Z"), 60.5)
        al = {"martin": "martin kakona"}
        self.assertEqual(normalize_name(" Martin ", al), normalize_name("Martin  Kakona", al))

    def test_load_intervals_merges_name_variants(self):
        csvtext = "annotator,date,start_utc,end_utc,label,source  \n" \
                  "Martin,2020-10-15,2020-10-15T00:00:10Z,2020-10-15T00:00:20Z,SAA,click  \n" \
                  "Martin Kakona,2020-10-15,2020-10-15T00:01:00Z,2020-10-15T00:01:10Z,SAA,click  \n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(csvtext)
        d = load_intervals(f.name, aliases={"martin": "martin kakona"})
        os.unlink(f.name)
        self.assertEqual(list(d), ["martin kakona"])
        self.assertEqual(len(d["martin kakona"]), 2)

    def test_mask_and_runs(self):
        times = list(range(0, 10))
        m = intervals_to_mask(times, [(2, 4), (4, 6)])
        self.assertEqual(m, [False, False, True, True, True, True, True, False, False, False])
        self.assertEqual(mask_to_intervals(times, m), [(2, 6)])
        self.assertEqual(intervals_to_mask(times, [(2, 4)], inclusive_end=False)[4], False)

    def test_sample_scores_hand_computed(self):
        times = list(range(10))
        ref = intervals_to_mask(times, [(0, 4)])   # 5 positives
        pred = intervals_to_mask(times, [(3, 7)])  # 5 positives, overlap 2 (t=3,4)
        s = sample_level_scores(ref, pred)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (2, 3, 3))
        self.assertAlmostEqual(s["f1"], 0.4)
        cov = [t < 5 for t in times]               # score only first 5 samples
        s2 = sample_level_scores(ref, pred, covered=cov)
        self.assertEqual((s2["tp"], s2["fp"], s2["fn"]), (2, 0, 3))

    def test_event_matching_tolerance_and_one_to_one(self):
        ref = [(1000, 1300)]
        self.assertEqual(match_events(ref, [(1030, 1330)], tol_s=60)["tp"], 1)       # 30 s shift
        far = match_events(ref, [(1500, 1800)], tol_s=60)                            # no overlap
        self.assertEqual((far["tp"], far["fp"], far["fn"]), (0, 1, 1))
        split = match_events(ref, [(1000, 1140), (1160, 1300)], tol_s=60)            # split pass
        self.assertEqual((split["tp"], split["fp"], split["fn"]), (1, 1, 0))
        m = match_events(ref, [(1090, 1360)], tol_s=60)["matches"][0]
        self.assertAlmostEqual(m[2], 1.5); self.assertAlmostEqual(m[3], 1.0)         # offsets in minutes

    def test_blocked_folds_buffer(self):
        times = list(range(100))
        for train, test in blocked_folds(times, 4, buffer_s=5):
            self.assertFalse(set(train) & set(test))
            lo, hi = min(test), max(test)
            self.assertTrue(all(i < lo - 5 or i > hi + 5 for i in train))

if __name__ == "__main__":
    unittest.main()
