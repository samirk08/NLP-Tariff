# Fall 2024 — U.S. Census tool automation

This phase pivoted from prototype models to a working end-to-end classification pipeline anchored on the official U.S. Census **3CE** tariff classification tool (`uscensus.prod.3ceonline.com`).

## What's here

| File | Role |
|---|---|
| `UROP_matching.py` | Selenium driver + GPT-4 decision loop. Walks the 3CE tool's interactive question tree, asks GPT to pick the most relevant option at each step (with composition tables filled in automatically), and returns the final HS code. |
| `1789_map_census.ipynb` | First attempt at running the 1789 schedule through the pipeline — early stress test of cross-century descriptions. |
| `hs_demo.ipynb` | Two-mode demo: text query → HS code, and image → GPT-4-vision description → HS code. |
| `my_app/` | Flask wrapper exposing the same pipeline through a small web UI (text and image upload). |

## Running it

The code expects an OpenAI API key in the environment:

```bash
export OPENAI_API_KEY=sk-...
python UROP_matching.py
```

ChromeDriver / Selenium are required for the Census-tool walk-through. Headless mode is the default.

## Why this approach was retired

The Census tool is authoritative for *contemporary* descriptions but was never designed for 1850-style language. Many historical line items walked the question tree to dead-ends. This is what motivated the all-embedding pipeline in `superurop/` — push the matching off the live web tool and onto offline embedding similarity.
