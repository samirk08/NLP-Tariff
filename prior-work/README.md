# Prior work — Spring 2024 algorithm tracks

This directory holds the earliest exploratory phase of the project. Five different approaches to HS-code prediction were prototyped in parallel and benchmarked on three reference years (1963, 1990, 1996) to learn which signal sources carried the most weight.

| Track | Idea | What it taught us |
|---|---|---|
| `sklearn-spacy/` | Classical NLP — TF-IDF / spaCy linguistic features into a linear classifier | Established a fast, deterministic baseline; floor for the deep models to beat |
| `huggingface-algorithm/` | Sentence-transformer embeddings + nearest-neighbor lookup | Pure-embedding match works well *within* a single year but degrades across decades |
| `gpt-algorithm/` | GPT-4 to rewrite/modernize descriptions before classification | LLM rewriting closed a real fraction of the cross-era language gap |
| `final-algorithm/` | **Hybrid** (HF embedding + GPT rewriting) — separate folders for 1963, 1990, 1996 | Hybrid beat either alone; gains came from combining surface-form and semantic signals — directly motivated the composite-similarity score in the SuperUROP pipeline |
| `picture-to-tariff/` | Image → GPT-4-vision description → HS code | Demonstrated end-to-end image classification as a separate use case |

**Status.** These scripts are preserved for the project record. They reflect the original development environment (some hard-coded paths in comments). They are not the supported entry point for new use — that is `superurop/` and `tariff-search/`.

**API keys.** Where the original scripts wrote `api_key = "your-api-key"` as a placeholder, those placeholders are kept. No real secrets are checked in.
