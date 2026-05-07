# Methodology

A longer-form write-up of the approach.

## The problem statement

A row in the 1789 U.S. tariff schedule looks like:

> *"Madeira wine, in casks; per gallon — 18 cents"*

A row in the 2023 schedule looks like:

> *2204.21.20.00 — Wine of fresh grapes, including fortified wines; grape must other than that of heading 2009: in containers holding 2 liters or less: of an alcoholic strength by volume not exceeding 14% vol., other.*

These describe the same product, but they share almost no overlapping vocabulary. Multiply this by ~250 years of intermediate schedules — each rewriting the taxonomy, splitting categories, merging others, or adopting new product classes — and exact-match approaches do not work.

We want a function `f(item_1789) → item_2023` that returns the best modern HS code for a given historical item, with a calibrated confidence score.

## Why we settled on year-by-year chaining

A single direct embedding match from 1789 to 2023 is unreliable:

1. **Vocabulary drift compounds.** The semantic distance between 1789 and 2023 descriptions is so large that even good embeddings often pick the wrong modern category.
2. **Information is lost.** Sometimes a 1789 item splits into multiple modern HS codes; sometimes a 2023 item consolidates several historical ones. A single jump cannot recover this fan-in / fan-out structure.
3. **Intermediate schedules are signposts.** Each adjacent-year pair has *much* less drift; matches are reliable when the gap is small.

So we **chain**: match each year *N* to the next year *N+1* with high confidence, and compose those edges into a path from 1789 → 2023. Errors are localised — a bad single-year match shows up as a low-confidence edge we can flag and review.

## The composite similarity score

A single similarity metric is brittle:

- **Cosine on transformer embeddings** gets meaning right but can be confused by paraphrase distance — *"sugar, refined"* and *"refined sugar"* score lower than they should.
- **Jaccard on tokens** rewards literal vocabulary overlap, which is exactly what fails across centuries.
- **Levenshtein on raw strings** rewards near-identical wording — useful when only minor edits separate adjacent years.

Combining them with weights (α=0.7, β=0.2, γ=0.1) gave the best validation accuracy on the held-out per-year benchmark.

## Bidirectional reconciliation

For every adjacent pair (*N*, *N+1*) we run the matcher in both directions independently:

- *forward:*  for each item in *N+1*, find the best match in *N*
- *reverse:*  for each item in *N*, find the best match in *N+1*

Where the two directions form a closed cycle (item A in *N* matches B in *N+1* and B matches back to A), confidence is highest. Where they don't agree, we keep the higher-scoring direction and tag the conflict for review. This is in `notebooks/06_combine_N_N1.ipynb`.

## What we don't claim

- The final mapping is **research-grade**, not authoritative. We're confident on coverage and on the consistency of the methodology, but per-row correctness still requires expert review for high-stakes use.
- The composite-similarity weights are tuned, not learned. A learned weight scheme — perhaps a small classifier on top of the three signals — is a clear next step.
- Items that genuinely have no modern counterpart (obsolete goods, pre-industrial categories) are mapped to the *closest* modern HS code anyway. There is no "no-match" output class. Era summaries in `notebooks/10_final_mapping_analysis.ipynb` partially address this.
