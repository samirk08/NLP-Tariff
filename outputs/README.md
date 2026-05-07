# Year-pair mapping outputs

This directory holds the raw outputs of the matching pipeline:

| Sub-directory | Direction | Size | What's inside |
|---|---|---|---|
| `N_N1_mappings/` | year *N* → year *N+1* (forward) | ~22 MB | 95 CSVs — one per adjacent-year pair from 1789 to 2023 |
| `N1_N_mappings/` | year *N+1* → year *N* (reverse) | ~68 MB | Same 95 pairs, opposite direction |
| `new_N_N1_mappings/` | forward, post-reconciliation | ~9 MB | After bidirectional cleanup in `notebooks/06_combine_N_N1.ipynb` |
| `new_N1_N_mappings/` | reverse, post-reconciliation | ~52 MB | Same as above, opposite direction |

Each CSV row is one item from the source year matched to its best counterpart in the target year, with the four similarity scores (cosine / jaccard / levenshtein / combined) attached.

These files are **not checked into Git** — they total ~150 MB. To regenerate them, run `notebooks/03_modify_mappings.ipynb` end-to-end (after `01_make_embeddings.ipynb` has cached every year's embeddings).
