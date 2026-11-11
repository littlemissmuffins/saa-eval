# Changelog

## v0.3
Prompted by an independent review of v0.2 by Lissandra Kruse Fuganti (matching selector and input checks).

- `match_events` now raises `ValueError` for a negative, NaN or infinite `tol_s`, and for reversed or non finite intervals. Zero length intervals are still allowed. Valid inputs give identical results.
- `compare_marks.py` has `--matching {greedy,maximum}` (default greedy) and prints the mode used.
- 3 new tests (21 in total). Still open: tie breaking by overlap inside the maximum mode (offered by Abhi Singh).

## v0.2
Patch by Abhi Singh, applied from the reviewed proposal against 96993dc. Only synthetic tests were run for this patch.

- Input validation: unequal mask lengths raise `ValueError`; times must be finite and non decreasing (equal timestamps allowed).
- `mask_to_intervals(times, mask, max_gap_s=None)`: with a value, a run splits when consecutive positive samples are more than `max_gap_s` apart. Default behaviour is unchanged.
- `match_events(..., matching="greedy" | "maximum")`: new maximum cardinality option (augmenting paths). It does not maximise total overlap among equal size matchings. Default unchanged.
- `blocked_folds(..., event_intervals=None)`: new event purged mode; the docstring no longer claims time buffer only folds cannot leak. New validation of `n_folds`, `buffer_s` and events.
- `compare_marks.py`: `--ref` and `--pred` use name normalisation; an unknown name exits with code 2 and lists annotators.
- Tests: original 6 unchanged, plus 12 regression tests (18 total).

## v0.1
Initial version (96993dc).
