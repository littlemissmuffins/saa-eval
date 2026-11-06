# saa-eval

Small, unit-tested scoring helpers for SAA interval annotations, written as an independent cross-check
for the NASA OSDR "LEO Dosimetry - SAA/GCR Separation Using ML" project.

## Conventions
- UTC times; intervals are [start, end] with an inclusive end by default.
- Statistics use a continuous time axis, never calendar days.
- Event matching is one-to-one. A pair is a candidate if the prediction overlaps the reference widened by a tolerance (default 60 s, because SPACEDOS timestamps are aligned to about one minute). This is not a test that both boundaries are within the tolerance, so boundary offsets are reported separately.
- `match_events(..., matching="greedy")` (default) takes pairs by largest overlap. `matching="maximum"` gives a deterministic maximum-cardinality matching. It maximises the number of matches only, not the total overlap among matchings of equal size.
- Name variants are normalised (for example "Martin" and "Martin Kakona" are the same annotator).

## Files
- `saa_eval.py`: parsing, masks, sample-level scores, event matching, blocked folds.
- `test_saa_eval.py`: unit tests (run `python3 -m unittest test_saa_eval -v`).
- `compare_marks.py`: command line comparison of two sets of marks.

## Quick start
```
python3 -m unittest test_saa_eval -v
python3 compare_marks.py kakona_reference_marks.csv --ref "martin kakona" --pred "some volunteer"
```

## Changes in v0.2
Review and patch by Abhi Singh (MIT licence kept). See `CHANGELOG.md`. Inputs that were silently mis-scored before (unequal lengths, unsorted or non-finite times, invalid fold requests) now raise `ValueError`. Results for valid inputs with default arguments are unchanged, and the original six tests pass unmodified. 18 tests in total.

## Provenance
Prepared with AI assistance (Claude) and reviewed, tested and run by the repository owner. Please report any definition that looks wrong.

## Limits
Not validated against the NASA SRAG windows yet. Time buffer only folds can leak a pass longer than the buffer; use `event_intervals` for event purged folds. Numbers should be cross-checked against other implementations.
