# NLP-Tariff

> **NLP for historical U.S. tariff classification (1789 – 2023)**

A multi-year MIT UROP / SuperUROP research project on mapping over two centuries of U.S. tariff descriptions to modern Harmonized System (HS) codes using transformer embeddings, composite similarity scoring, and large-language-model assisted classification.

---

## Why this is hard

U.S. tariff schedules from 1789 onward describe goods in language that has drifted dramatically over time. *"Cotton cloth, fine, dyed, of British manufacture"* in 1850 has no direct counterpart in the modern HS taxonomy. Joining this historical record to today's classification is a pre-requisite for any long-run trade economics work — but doing it by hand for 250 years and tens of thousands of line items is intractable.

The goal of this project is to build that bridge automatically.

---

## Repository structure

```
NLP-Tariff/
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── requirements.txt
├── .gitignore
│
├── prior-work/                ← Spring 2024: first experiments — five algorithm tracks
│   ├── sklearn-spacy/             linear-model + spaCy baseline
│   ├── huggingface-algorithm/     transformer embeddings (all-MiniLM)
│   ├── gpt-algorithm/             GPT-4 description rewriting + classification
│   ├── final-algorithm/           hybrid (HF + GPT) per-year studies (1963, 1990, 1996)
│   └── picture-to-tariff/         vision: image → description → HS code
│
├── fall2024/                  ← Fall 2024: U.S. Census-tool automation pipeline
│   ├── UROP_matching.py           Selenium + GPT-4 driver for uscensus.prod.3ceonline.com
│   ├── 1789_map_census.ipynb      mapping the 1789 tariff schedule via the Census tool
│   ├── hs_demo.ipynb              text + image → HS code demo
│   └── my_app/                    Flask web prototype
│
├── superurop/                 ← Spring 2025 SuperUROP: production embedding pipeline
│   ├── src/
│   │   ├── process_years.py           year-by-year embedding-based matching
│   │   ├── process_years_N_N1.py      forward-direction with synthesised keys
│   │   ├── process_years_N_N1_COP.py  parameter-tuning variant
│   │   └── census_algo.py             refactored Census-tool driver
│   ├── notebooks/                     11 numbered notebooks: embed → match → reconcile → analyse
│   └── outputs/                       year-pair mapping CSVs (1789 → 2023, both directions)
│
├── tariff-search/             ← Standalone Python package: query the final mapping
│   ├── tariff_search/                 Faiss-backed semantic search
│   ├── setup.py
│   └── README.md                      package-specific docs
│
├── data/
│   ├── 1789HS.xlsx                    1789 baseline schedule
│   ├── UAR_era_summary.csv            era-level mapping summary
│   └── samples/                       small data samples (full data not redistributed)
│
└── docs/
    └── methodology.md                 detailed write-up of approach
```

---

## Methodology — the SuperUROP pipeline

The SuperUROP work is the production system. It chains four stages:

**1. Embed every tariff description, every year.**
Each year's tariff schedule is encoded with `sentence-transformers/all-mpnet-base-v2` (768-dim) and cached. Notebook: `superurop/notebooks/01_make_embeddings.ipynb`.

**2. Match adjacent years with a composite similarity score.**
For every consecutive pair (year *N*, year *N+1*) and every item in *N+1*, we score candidates in *N* by a weighted blend:

```
similarity = α · cosine(emb_N1, emb_N) + β · jaccard(tok_N1, tok_N) + γ · levenshtein(desc_N1, desc_N)
                with α = 0.7, β = 0.2, γ = 0.1
```

Cosine captures meaning, Jaccard the shared-token structure, Levenshtein the surface form — none alone is enough across two centuries of language drift. Implementation: `superurop/src/process_years.py`.

**3. Reconcile bidirectional matches.**
We run the matcher both ways (*N → N+1* and *N+1 → N*) and combine. Where the two directions agree, confidence is high; where they disagree, we keep the stronger composite score and flag the conflict. Notebook: `superurop/notebooks/06_combine_N_N1.ipynb`.

**4. Chain the per-year mappings into a 1789 → 2023 lookup.**
Year-by-year edges compose into long chains. The final mapping artifact links every 1789 line item to its 2023 HS-code descendant.

Per-year mapping CSVs are checked in under `superurop/outputs/`. The final 1789 → 2023 mapping pickle is too large for Git (≈400 MB) and is regenerated from the notebooks.

---

## Project arc

| Era | What changed | Key artifact |
|---|---|---|
| **Spring 2024** (`prior-work/`) | Tried five different approaches in parallel — sklearn, HF embeddings, GPT, hybrid, vision — on individual benchmark years (1963, 1990, 1996) | F1 score comparisons across approaches |
| **Fall 2024** (`fall2024/`) | Built a real classification pipeline: Selenium driving the U.S. Census 3CE tool with GPT-4 making decisions; first attempt at the 1789 schedule | `UROP_matching.py`, Flask web demo |
| **Spring 2025 — SuperUROP** (`superurop/`) | Pivoted from web-scraping to a fully embedding-based, fully automated year-by-year pipeline — scaled to 1789 – 2023 | `process_years.py`, year-pair mapping CSVs |
| **Spring 2025 — Package** (`tariff-search/`) | Productised the final mapping into a queryable Faiss-indexed Python library | `tariff_search.TariffSearch` |

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

### Reproduce the SuperUROP pipeline

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # only needed for the Census-tool / GPT-assisted code
export HF_TOKEN=...         # optional; the embedding model is public
```

Run the notebooks in numerical order from `superurop/notebooks/`:
1. `01_make_embeddings.ipynb` – encode every year's schedule
2. `03_modify_mappings.ipynb` → `05_modify_1989_2023.ipynb` – build year-pair mappings
3. `06_combine_N_N1.ipynb` – bidirectional reconciliation
4. `07_test_mappings.ipynb` / `08_test_mappings_N_N1.ipynb` – sanity-check coverage
5. `10_final_mapping_analysis.ipynb` – end-to-end analysis and figures

The raw `all_tariffs.csv` and the precomputed `embeddings/` directory are not redistributed in the repo (~hundreds of MB); see `data/README.md` for sourcing.

---

## Acknowledgements

This work was carried out at MIT under the UROP and SuperUROP programs. Thanks to the supervising research group for direction and to the broader MIT Tradelab ecosystem for trade-data infrastructure context.

## License

MIT — see [`LICENSE`](LICENSE).
