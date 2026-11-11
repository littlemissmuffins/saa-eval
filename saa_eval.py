"""Scoring helpers for SAA interval annotations (independent cross-check implementation).

Conventions (document these in any report that uses the numbers):
- Times are seconds since the Unix epoch, UTC. Intervals are [start, end] with an inclusive end
  unless inclusive_end=False.
- Statistics are computed on a continuous time axis, never per calendar day.
- Event matching is one-to-one: a reference pass can match at most one predicted pass, so a pass
  split into two predictions gives one match and one false positive.
- tol_s (about 60 s, because SPACEDOS timestamps are only aligned to about one minute) widens the
  reference interval on both sides; any overlap with the widened reference makes a candidate
  pair. It is NOT a "both boundaries within tol_s" test: report start/end offsets separately.
- Sample times passed to the mask/fold helpers must be finite and sorted ascending (ValueError
  otherwise), and masks/coverage must have the same length as times.
"""
import csv
import math
from datetime import datetime, timezone


def _check_times(times):
    """Raise ValueError unless times are all finite and non-decreasing."""
    prev = None
    for i, t in enumerate(times):
        if not math.isfinite(t):
            raise ValueError(f"non-finite sample time at index {i}: {t!r}")
        if prev is not None and t < prev:
            raise ValueError(f"sample times not sorted ascending at index {i}: {t!r} < {prev!r}")
        prev = t


def _check_nonneg(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


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
    _check_times(times)
    mask = [False] * len(times)
    ivs = merge_intervals(intervals)
    j = 0
    for i, t in enumerate(times):  # times must be sorted ascending
        while j < len(ivs) and ivs[j][1] < t if inclusive_end else j < len(ivs) and ivs[j][1] <= t:
            j += 1
        if j < len(ivs) and ivs[j][0] <= t and (t <= ivs[j][1] if inclusive_end else t < ivs[j][1]):
            mask[i] = True
    return mask


def mask_to_intervals(times, mask, max_gap_s=None):
    """Run-length encode a boolean mask into (start_time, end_time) intervals.

    By default (max_gap_s=None, the historical behaviour) consecutive True samples are joined no
    matter how far apart their times are, so a run continues across missing observations. Callers
    must either pass a justified max_gap_s (a run is split when two consecutive positive samples
    are more than max_gap_s apart) or pre-segment the data at known gaps.
    """
    if len(times) != len(mask):
        raise ValueError(f"times and mask lengths differ: {len(times)} != {len(mask)}")
    _check_times(times)
    if max_gap_s is not None:
        _check_nonneg("max_gap_s", max_gap_s)
    out, start = [], None
    for t, m in zip(times, mask):
        if m and start is not None and max_gap_s is not None and t - last > max_gap_s:
            out.append((start, last))
            start = None
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
    unmarked samples inside it count as negatives and samples outside it are ignored.
    All lists must have the same length (ValueError otherwise)."""
    if len(ref_mask) != len(pred_mask):
        raise ValueError(f"ref_mask and pred_mask lengths differ: {len(ref_mask)} != {len(pred_mask)}")
    if covered is not None and len(covered) != len(ref_mask):
        raise ValueError(f"covered length {len(covered)} != mask length {len(ref_mask)}")
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


def _max_cardinality(adj, n_pred):
    """Maximum-cardinality bipartite matching by augmenting paths (Kuhn), iterative DFS.

    adj[i] lists candidate pred indices for ref i in ascending order; refs are processed in index
    order, so the result is deterministic. Returns [(ref_i, pred_j), ...]. Maximises the NUMBER of
    pairs only, not total overlap.
    """
    match_p = [-1] * n_pred
    match_r = [-1] * len(adj)
    for i in range(len(adj)):
        visited = [False] * n_pred
        reached_from = {}
        stack, found = [[i, 0]], -1
        while stack:
            frame = stack[-1]
            r, k = frame
            if k >= len(adj[r]):
                stack.pop()
                continue
            frame[1] += 1
            j = adj[r][k]
            if visited[j]:
                continue
            visited[j] = True
            reached_from[j] = r
            if match_p[j] == -1:
                found = j
                break
            stack.append([match_p[j], 0])
        j = found
        while j != -1:  # augment along the path back to ref i
            r = reached_from[j]
            prev = match_r[r]
            match_p[j], match_r[r] = r, j
            j = -1 if r == i else prev
    return [(i, j) for i, j in enumerate(match_r) if j != -1]


def match_events(ref, pred, tol_s=60.0, matching="greedy"):
    """One-to-one event matching. A pair is a candidate if the intervals overlap after widening
    the reference by tol_s on both sides (a zero-length touch of the widened edge also counts).
    This is not a "both boundaries within tol_s" criterion.

    matching="greedy" (default, historical): pairs are taken greedily by largest overlap. This can
      return fewer matches than possible (a long prediction can strand another reference).
    matching="maximum": deterministic maximum-cardinality matching over the same candidate pairs
      (augmenting paths, refs and preds traversed in index order). It maximises the number of
      matches; among equal-size matchings it does NOT maximise total overlap.

    Returns dict with matches [(ref_i, pred_j, start_offset_min, end_offset_min)], event counts
    and precision/recall/F1. Offsets are pred minus ref (positive = later).
    """
    if matching not in ("greedy", "maximum"):
        raise ValueError(f"matching must be 'greedy' or 'maximum', got {matching!r}")
    if not (isinstance(tol_s, (int, float)) and math.isfinite(tol_s) and tol_s >= 0):
        raise ValueError(f"tol_s must be a finite number >= 0, got {tol_s!r}")
    for label, events in (("ref", ref), ("pred", pred)):
        for k, ev in enumerate(events):
            if len(ev) != 2 or not (math.isfinite(ev[0]) and math.isfinite(ev[1])) or ev[0] > ev[1]:
                raise ValueError(f"{label}[{k}] must be a finite (start, end) with start <= end, got {tuple(ev)!r}")
    cands = []
    for i, r in enumerate(ref):
        widened = (r[0] - tol_s, r[1] + tol_s)
        for j, p in enumerate(pred):
            ov = _overlap(widened, p)
            if ov > 0 or (widened[0] <= p[0] <= widened[1]) or (p[0] <= widened[0] <= p[1]):
                cands.append((ov, i, j))
    if matching == "maximum":
        adj = [[] for _ in ref]
        for _, i, j in cands:  # cands are generated in (i, j) ascending order
            adj[i].append(j)
        pairs = _max_cardinality(adj, len(pred))
    else:
        cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        used_r, used_p, pairs = set(), set(), []
        for ov, i, j in cands:
            if i in used_r or j in used_p:
                continue
            used_r.add(i)
            used_p.add(j)
            pairs.append((i, j))
    matches = [(i, j, (pred[j][0] - ref[i][0]) / 60.0, (pred[j][1] - ref[i][1]) / 60.0)
               for i, j in pairs]
    scores = prf(len(matches), len(pred) - len(matches), len(ref) - len(matches))
    scores["matches"] = sorted(matches)
    return scores


def blocked_folds(times, n_folds, buffer_s, event_intervals=None):
    """Time-ordered folds for ML evaluation. Each fold is a contiguous block of the sorted
    timeline. Returns [(train_indices, test_indices), ...].

    Time-buffer-only mode (event_intervals=None, historical): training samples within buffer_s of
    the test time block are dropped. This does NOT prevent leakage of a pass longer than buffer_s
    that crosses a fold edge: its far part can still land in training.

    Event-purged mode: event_intervals is a list of COMPLETE (start, end) events (e.g. merged
    reference passes, not clipped to any fold). For every event touching the test time block
    [t_lo, t_hi], training samples inside [event_start - buffer_s, event_end + buffer_s] are also
    dropped, so no part of such an event (plus buffer) is in training. Leakage protection is only
    as good as the supplied events' completeness.

    Requires finite sorted times, finite buffer_s >= 0 and 1 <= n_folds <= len(times).
    """
    _check_times(times)
    _check_nonneg("buffer_s", buffer_s)
    n = len(times)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 1 <= n_folds <= n:
        raise ValueError(f"n_folds must be an integer with 1 <= n_folds <= len(times)={n}, got {n_folds!r}")
    events = []
    if event_intervals is not None:
        for s, e in event_intervals:
            if not (math.isfinite(s) and math.isfinite(e)) or e < s:
                raise ValueError(f"invalid event interval: {(s, e)!r}")
            events.append((s, e))
    folds = []
    for k in range(n_folds):
        lo, hi = k * n // n_folds, (k + 1) * n // n_folds
        t_lo, t_hi = times[lo], times[hi - 1]
        test = list(range(lo, hi))
        windows = [(t_lo - buffer_s, t_hi + buffer_s)]
        windows += [(s - buffer_s, e + buffer_s) for s, e in events if s <= t_hi and e >= t_lo]
        train = [i for i in range(n) if not (lo <= i < hi)
                 and not any(a <= times[i] <= b for a, b in windows)]
        folds.append((train, test))
    return folds


