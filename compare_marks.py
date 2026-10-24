"""Compare two sets of SAA interval marks (for example two volunteers, or a model against a human).

Usage:
  python3 compare_marks.py marks.csv --ref "martin kakona" --pred "hitaeshi sehgal" --tol-s 60
  python3 compare_marks.py ref_marks.csv --pred-file model_marks.csv --ref "martin kakona" --pred "model"

Both CSVs need columns whose names contain 'annot', 'start' and 'end' (UTC ISO times).
"""
import argparse
import statistics
from saa_eval import load_intervals, match_events

ALIASES = {"martin": "martin kakona"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_csv")
    ap.add_argument("--pred-file", help="second CSV; default is to use the same file as ref_csv")
    ap.add_argument("--ref", required=True, help="reference annotator name")
    ap.add_argument("--pred", required=True, help="predicted/compared annotator name")
    ap.add_argument("--tol-s", type=float, default=60.0)
    a = ap.parse_args()
    ref_all = load_intervals(a.ref_csv, ALIASES)
    pred_all = load_intervals(a.pred_file or a.ref_csv, ALIASES)
    ref = ref_all[a.ref.casefold()]
    pred = pred_all[a.pred.casefold()]
    res = match_events(ref, pred, tol_s=a.tol_s)
    print(f"reference passes: {len(ref)}  compared passes: {len(pred)}  tolerance: +/-{a.tol_s:.0f} s")
    print(f"matched: {res['tp']}  extra (false positive): {res['fp']}  missed (false negative): {res['fn']}")
    print(f"precision {res['precision']:.3f}  recall {res['recall']:.3f}  F1 {res['f1']:.3f}")
    if res["matches"]:
        starts = [m[2] for m in res["matches"]]
        ends = [m[3] for m in res["matches"]]
        print(f"start offset (min), median {statistics.median(starts):+.2f}, range {min(starts):+.2f} to {max(starts):+.2f}")
        print(f"end offset (min),   median {statistics.median(ends):+.2f}, range {min(ends):+.2f} to {max(ends):+.2f}")
    print("Offsets are compared minus reference (positive = later).")


if __name__ == "__main__":
    main()
