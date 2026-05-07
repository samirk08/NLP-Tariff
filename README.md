# NLP-Tariff

> **NLP for historical U.S. tariff classification (1789 – 2023)**

A research project on mapping over two centuries of U.S. tariff line items to modern Harmonized System (HS) codes using transformer embeddings and a composite similarity score that combines semantic, token-overlap, and surface-form signals.

---

## Why this is hard

U.S. tariff schedules from 1789 onward describe goods in language that has drifted dramatically over time. *"Cotton cloth, fine, dyed, of British manufacture"* in 1850 has no direct counterpart in the modern HS taxonomy. Joining this historical record to today's classification is a pre-requisite for any long-run trade economics work — but doing it by hand for 250 years and tens of thousands of line items is intractable.

This project builds that bridge automatically.

---

## How it works

The pipeline chains four stages:

**1. Embed every tariff description, every year.**
Each year's schedule is encoded with `sentence-transformers/all-mpnet-base-v2` (768-dim) and cached. → `notebooks/01_make_embeddings.ipynb`

**2. Match adjacent years with a composite similarity score.**
For every consecutive pair (year *N*, year *N+1*) and every item in *N+1*, we score candidates in *N* by a weighted blend:

```
similarity = α · cosine(emb_N1, emb_N) + β · jaccard(tok_N1, tok_N) + γ · levenshtein(desc_N1, desc_N)
                with α = 0.7, β = 0.2, γ = 0.1
```

Cosine captures meaning; Jaccard captures shared-token structure; Levenshtein captures surface form. None alone is enough across two centuries of language drift. → `src/process_years.py`

**3. Reconcile bidirectional matches.**
Run the matcher both ways (*N → N+1* and *N+1 → N*). Where the two directions agree, confidence is high; where they disagree, keep the stronger composite score and flag the conflict. → `notebooks/06_combine_N_N1.ipynb`

**4. Chain per-year edges into a 1789 → 2023 lookup.**
Year-by-year mappings compose into long chains. The final mapping links every 1789 line item to its 2023 HS-code descendant. → `notebooks/10_final_mapping_analysis.ipynb`

For the long-form write-up of the methodology, see [`docs/methodology.md`](docs/methodology.md).

---

## Repository structure

```
NLP-Tariff/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                          ← matching pipeline
│   ├── process_years.py              core matcher (composite-similarity, parallel)
│   ├── process_years_N_N1.py         variant that synthesises a key for items
│   │                                 missing an HS code
│   └── process_years_N_N1_COP.py     parameter-tuning variant
│
├── notebooks/                    ← end-to-end pipeline, in order
│   ├── 01_make_embeddings.ipynb
│   ├── 02_embedd_digits.ipynb
│   ├── 03_modify_mappings.ipynb
│   ├── 04_modify_1963_1988.ipynb
│   ├── 05_modify_1989_2023.ipynb
│   ├── 06_combine_N_N1.ipynb
│   ├── 07_test_mappings.ipynb
│   ├── 08_test_mappings_N_N1.ipynb
│   ├── 09_similarity_for_2023.ipynb
│   ├── 10_final_mapping_analysis.ipynb
│   └── 11_user_query.ipynb
│
├── outputs/                      ← year-pair mapping CSVs (1789 → 2023, both directions)
│
├── tariff-search/                ← standalone Python package: query the final mapping
│   ├── tariff_search/                Faiss-backed semantic search
│   ├── setup.py
│   └── README.md
│
├── data/
│   ├── 1789HS.xlsx                   1789 baseline schedule
│   ├── UAR_era_summary.csv           era-level mapping summary
│   └── samples/                      small annotated samples
│
└── docs/
    └── methodology.md                detailed methodology write-up
```

---

## Quick start

### Search the final mapping

```bash
cd tariff-search
pip install -e .
```

```python
from tariff_search import TariffSearch

searcher = TariffSearch(data_dir="path/to/prepared/data")
results = searcher.search("cotton cloth, bleached", top_k=5)
print(results[["combined_similarity", "Description_N", "HS_N"]])
```

Full package docs: [`tariff-search/README.md`](tariff-search/README.md).

### Reproduce the pipeline

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # optional; the embedding model is public
```

Run the notebooks in numerical order from `notebooks/`:
1. `01_make_embeddings.ipynb` – encode every year's schedule
2. `03_modify_mappings.ipynb` → `05_modify_1989_2023.ipynb` – build year-pair mappings
3. `06_combine_N_N1.ipynb` – bidirectional reconciliation
4. `07_test_mappings.ipynb` / `08_test_mappings_N_N1.ipynb` – sanity-check coverage
5. `10_final_mapping_analysis.ipynb` – end-to-end analysis and figures

The raw `all_tariffs.csv` and the precomputed `embeddings/` directory are not redistributed in the repo; see [`data/README.md`](data/README.md) for sourcing.

---

## License

MIT — see [`LICENSE`](LICENSE).
