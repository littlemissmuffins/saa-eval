# saa-eval

Small, unit-tested scoring helpers for SAA interval annotations, written as an independent cross-check
for the NASA OSDR "LEO Dosimetry - SAA/GCR Separation Using ML" project.

## Conventions
- UTC times; intervals are [start, end] with an inclusive end by default.
- Statistics use a continuous time axis, never calendar days.
- Event matching is one-to-one with a boundary tolerance (default 60 s, because SPACEDOS timestamps are aligned to about one minute).
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

## Provenance
Prepared with AI assistance (Claude) and reviewed, tested and run by the repository owner. Please report any definition that looks wrong.

## Limits
Not validated against the NASA SRAG windows yet. Numbers should be cross-checked against other implementations.
