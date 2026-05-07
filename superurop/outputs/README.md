# Year-pair mapping outputs

This directory holds the raw outputs of the SuperUROP matching pipeline:

| Sub-directory | Direction | Size | What's inside |
|---|---|---|---|
| `N_N1_mappings/` | year *N* → year *N+1* (forward) | ~22 MB | 95 CSVs — one per adjacent-year pair from 1789 to 2023 |
| `N1_N_mappings/` | year *N+1* → year *N* (reverse) | ~68 MB | Same 95 pairs, opposite direction |
| `new_N_N1_mappings/` | forward, post-reconciliation | ~9 MB | After bidirectional cleanup in `06_combine_N_N1.ipynb` |
| `new_N1_N_mappings/` | reverse, post-reconciliation | ~52 MB | Same as above, opposite direction |

Each CSV row is one item from the source year matched to its best counterpart in the target year, with the four similarity scores (cosine / jaccard / levenshtein / combined) attached.

These files are **not yet checked into Git** — they total ~150 MB and the source disk was near-full at repo-creation time, so the bulk transfer is staged separately. To populate locally, run from the source machine:

```bash
ditto "../UROP 6.UAR resources/N_N1_mappings"     N_N1_mappings
ditto "../UROP 6.UAR resources/N1_N_mappings"     N1_N_mappings
ditto "../UROP 6.UAR resources/new_N_N1_mappings" new_N_N1_mappings
ditto "../UROP 6.UAR resources/new_N1_N_mappings" new_N1_N_mappings
```

To regenerate from scratch, run `superurop/notebooks/03_modify_mappings.ipynb` end-to-end.
