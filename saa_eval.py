"""Scoring helpers for SAA interval annotations (independent cross-check implementation).

Conventions (document these in any report that uses the numbers):
- Times are seconds since the Unix epoch, UTC. Intervals are [start, end] with an inclusive end
  unless inclusive_end=False.
- Statistics are computed on a continuous time axis, never per calendar day.
- Event matching is one-to-one: a reference pass can match at most one predicted pass, so a pass
  split into two predictions gives one match and one false positive.
- tol_s is the boundary tolerance (about 60 s, because SPACEDOS timestamps are only aligned to
  about one minute).
"""
import csv
from datetime import datetime, timezone


def parse_ts(text):
    """Parse ISO 8601 UTC text such as 2020-10-20T02:07:31.100Z into epoch seconds."""
    text = text.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def normalize_name(name, aliases=None):
    """Case-fold and collapse spaces; apply an alias map (e.g. {'martin': 'martin kakona'})."""
    key = " ".join(name.strip().split()).casefold()
    return (aliases or {}).get(key, key)


def load_intervals(path, aliases=None):
    """Read a marks CSV with columns containing 'annotator', 'start' and 'end' in their names.

    Returns {annotator: [(start_s, end_s), ...]} sorted by start. Header spaces are stripped.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        cols = {h.lower(): h for h in reader.fieldnames}
        find = lambda token: next(cols[c] for c in cols if token in c)
        a_col, s_col, e_col = find("annot"), find("start"), find("end")
        out = {}
        for row in reader:
            who = normalize_name(row[a_col], aliases)
            s, e = parse_ts(row[s_col]), parse_ts(row[e_col])
            if e < s:
                raise ValueError(f"end before start in row: {row}")
            out.setdefault(who, []).append((s, e))
    return {k: sorted(v) for k, v in out.items()}


def merge_intervals(intervals, gap_s=0.0):
    """Merge overlapping or touching intervals (within gap_s)."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def intervals_to_mask(times, intervals, inclusive_end=True):
    """Boolean array: True where a sample time falls inside any interval."""
    mask = [False] * len(times)
    ivs = merge_intervals(intervals)
    j = 0
    for i, t in enumerate(times):  # times must be sorted ascending
        while j < len(ivs) and ivs[j][1] < t if inclusive_end else j < len(ivs) and ivs[j][1] <= t:
            j += 1
        if j < len(ivs) and ivs[j][0] <= t and (t <= ivs[j][1] if inclusive_end else t < ivs[j][1]):
            mask[i] = True
    return mask


def mask_to_intervals(times, mask):
    """Run-length encode a boolean mask into (start_time, end_time) intervals."""
    out, start = [], None
    for t, m in zip(times, mask):
        if m and start is None:
            start, last = t, t
        elif m:
            last = t
        elif start is not None:
            out.append((start, last))
            start = None
    if start is not None:
        out.append((start, last))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def sample_level_scores(ref_mask, pred_mask, covered=None):
    """Sample-level scores. covered (bool list) restricts scoring to the annotated span, so
    unmarked samples inside it count as negatives and samples outside it are ignored."""
    tp = fp = fn = 0
    for i, (r, p) in enumerate(zip(ref_mask, pred_mask)):
        if covered is not None and not covered[i]:
            continue
        tp += r and p
        fp += (not r) and p
        fn += r and (not p)
    return prf(tp, fp, fn)


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(ref, pred, tol_s=60.0):
    """One-to-one event matching. A pair is a candidate if the intervals overlap after widening
    the reference by tol_s on both sides. Pairs are taken greedily by largest overlap.

    Returns dict with matches [(ref_i, pred_j, start_offset_min, end_offset_min)], event counts
    and precision/recall/F1. Offsets are pred minus ref (positive = later).
    """
    cands = []
    for i, r in enumerate(ref):
        widened = (r[0] - tol_s, r[1] + tol_s)
        for j, p in enumerate(pred):
            ov = _overlap(widened, p)
            if ov > 0 or (widened[0] <= p[0] <= widened[1]) or (p[0] <= widened[0] <= p[1]):
                cands.append((ov, i, j))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_r, used_p, matches = set(), set(), []
    for ov, i, j in cands:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        matches.append((i, j, (pred[j][0] - ref[i][0]) / 60.0, (pred[j][1] - ref[i][1]) / 60.0))
    scores = prf(len(matches), len(pred) - len(matches), len(ref) - len(matches))
    scores["matches"] = sorted(matches)
    return scores


def blocked_folds(times, n_folds, buffer_s):
    """Time-ordered folds for ML evaluation. Each fold is a contiguous block of the sorted
    timeline; training samples within buffer_s of the test block are dropped so passes cut by
    a fold edge cannot leak. Returns [(train_indices, test_indices), ...]."""
    n = len(times)
    folds = []
    for k in range(n_folds):
        lo, hi = k * n // n_folds, (k + 1) * n // n_folds
        t_lo, t_hi = times[lo], times[hi - 1]
        test = list(range(lo, hi))
        train = [i for i in range(n) if not (lo <= i < hi) and not (t_lo - buffer_s <= times[i] <= t_hi + buffer_s)]
        folds.append((train, test))
    return folds
