# SuperUROP — Spring 2025 production pipeline

The main research artifact. A fully automated, embedding-based pipeline that maps every tariff line item from 1789 through 2023 to its closest counterpart in the next year, then chains those edges into a 1789 → 2023 lookup.

## Pipeline overview

```
            ┌──────────────────────┐
            │  raw tariff CSVs     │
            │  (one per year)      │
            └─────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  01_make_embeddings.ipynb  │   all-mpnet-base-v2 (768d)
         │  → embeddings/{year}.pkl   │
         └────────────┬───────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ src/process_years.py            │  α·cos + β·jaccard + γ·lev
        │ + 03/04/05_modify_*.ipynb       │  (α=0.7, β=0.2, γ=0.1)
        │ → outputs/N_N1_mappings/*.csv   │
        │   (year N → year N+1, per pair) │
        └────────────┬────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐    same as above, reverse direction
         │ outputs/N1_N_mappings/...  │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │ 06_combine_N_N1.ipynb      │  bidirectional reconciliation
         │ → outputs/new_*_mappings/  │  high-confidence year-pair edges
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │ 10_final_mapping_analysis  │  chain edges, summarise eras
         │ → FINAL_MAPPING_*.pkl      │  (≈400 MB; not in repo, regenerate)
         └────────────────────────────┘
```

## Folder map

| Path | What's in it |
|---|---|
| `src/process_years.py` | Core matcher — composite-similarity ranking with parallel ThreadPoolExecutor |
| `src/process_years_N_N1.py` | Variant: synthesises a unique key for items missing an HS code |
| `src/process_years_N_N1_COP.py` | Parameter-tuning variant kept for reference |
| `src/census_algo.py` | Refactored Selenium + GPT driver from the Fall 2024 work |
| `notebooks/01_make_embeddings.ipynb` | Encode every year's schedule, cache to `embeddings/` |
| `notebooks/02_embedd_digits.ipynb` | Embedding of HS-code digit prefixes (used for tie-breaking) |
| `notebooks/03_modify_mappings.ipynb` | Apply matcher to all year pairs |
| `notebooks/04_modify_1963_1988.ipynb` | Refinement for the 1963–1988 era (TSUS → HS transition) |
| `notebooks/05_modify_1989_2023.ipynb` | Refinement for the modern HS era |
| `notebooks/06_combine_N_N1.ipynb` | Bidirectional reconciliation |
| `notebooks/07_test_mappings.ipynb` | Coverage / quality checks |
| `notebooks/08_test_mappings_N_N1.ipynb` | Same, for the *_N_N1 variant |
| `notebooks/09_similarity_for_2023.ipynb` | Score and store similarity for the 2023 endpoint |
| `notebooks/10_final_mapping_analysis.ipynb` | End-to-end analysis, era summaries, figures |
| `notebooks/11_user_query.ipynb` | Interactive query of the final mapping |
| `outputs/N_N1_mappings/` | Forward-direction year-pair mappings (95 CSVs) |
| `outputs/N1_N_mappings/` | Reverse-direction year-pair mappings |
| `outputs/new_N_N1_mappings/` | Reconciled, post-bidirectional |
| `outputs/new_N1_N_mappings/` | Reconciled, reverse |

## Outputs

The 95 CSVs in each `outputs/*_mappings/` directory are the per-year-pair mapping tables. Each row is one item from the source year matched to its best counterpart in the target year, with the four similarity scores (cosine / jaccard / levenshtein / combined).

The final 1789 → 2023 lookup dictionary is too large to redistribute via Git (~400 MB). Regenerate it from `notebooks/10_final_mapping_analysis.ipynb`.

## Environment

```bash
pip install -r ../requirements.txt
export HF_TOKEN=...        # optional; the embedding model loads anonymously
export OPENAI_API_KEY=...  # only needed if running census_algo.py
```
