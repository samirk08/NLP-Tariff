# Data

## Files in this directory

| File | What it is |
|---|---|
| `1789HS.xlsx` | The 1789 U.S. tariff schedule with hand-aligned HS-code seeds — the entry point for chaining forward through history. |
| `UAR_era_summary.csv` | One-row-per-era summary of the final mapping: counts, coverage, average similarity. |
| `samples/jerik_sample.csv` | Small annotated sample from the 1989 schedule used for spot-checking matches. |

## Files NOT redistributed via Git

These are too large or upstream-licensed; you'll need to source them yourself.

| File | Where it comes from | Used by |
|---|---|---|
| `all_tariffs.csv` | Concatenated raw tariff data, all years | `src/process_years.py`, all `modify_*.ipynb` |
| `embeddings/embeddings_{year}.pkl` | Output of `notebooks/01_make_embeddings.ipynb` (regenerable) | `process_years.py`, similarity notebooks |
| `FINAL_MAPPING_1789_2023.pkl` | Output of `notebooks/10_final_mapping_analysis.ipynb` (~400 MB) | `tariff-search/`, `notebooks/11_user_query.ipynb` |
| `df_with_embeddings.pkl` | Concatenated DataFrame + embeddings | `tariff-search/convert_existing_data.py` |

To regenerate everything from scratch, work through `notebooks/` in numerical order. The first run is compute-heavy (an embedding pass over every year's schedule); subsequent runs hit the cache.

## Source

Per-year tariff data was assembled from U.S. trade-statistics archives (USITC / Census historical schedules and modern HS releases).
