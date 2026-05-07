# Tariff Search

Fast and efficient search tool for finding similar tariff descriptions across historical US tariff data from 1789-2023.

## Features

- **Fast Search**: Uses optimized vector similarity search with optional Faiss support
- **Efficient Storage**: Separates embeddings from metadata for quick loading
- **Multi-Similarity Ranking**: Combines semantic, Jaccard, and Levenshtein similarities
- **Easy to Use**: Simple Python API for searching tariff descriptions

## Installation

```bash
pip install tariff-search
```

For GPU-accelerated search with Faiss:
```bash
pip install tariff-search[faiss-gpu]
```

## Quick Start

### 1. Prepare Your Data

First, convert your existing DataFrame with embeddings into the optimized format:

```python
from tariff_search import prepare_data_files

# Convert your large pickle file
prepare_data_files(
    input_pickle_path="df_with_embeddings.pkl",
    output_dir="tariff_search_data",
    embedding_column="description_embedding"
)
```

### 2. Search for Similar Tariffs

```python
from tariff_search import TariffSearch

# Initialize the search engine
searcher = TariffSearch(data_dir="tariff_search_data")

# Search for similar items
results = searcher.search(
    query="Cotton cloth, bleached, containing synthetic fiber",
    top_k=5
)

# Display results
for idx, row in results.iterrows():
    print(f"Similarity: {row['combined_similarity']:.3f}")
    print(f"Description: {row['Description_N']}")
    print(f"HS Code: {row['HS_N']}")
    print("-" * 50)
```

## API Reference

### TariffSearch

Main search class for finding similar tariff descriptions.

```python
TariffSearch(
    data_dir: str = None,          # Directory with prepared data
    model_name: str = '...',       # Transformer model for queries
    use_faiss: bool = True,        # Use Faiss for fast search
    device: str = None             # 'cuda' or 'cpu'
)
```

#### Methods

- `search(query, top_k=5, alpha=0.7, beta=0.2, gamma=0.1, rerank=True)`: Search for similar descriptions
- `batch_search(queries, **kwargs)`: Search multiple queries at once

### Data Preparation

```python
prepare_data_files(
    input_pickle_path: str,        # Path to DataFrame pickle
    output_dir: str,               # Output directory
    embedding_column: str = '...'  # Column with embeddings
)
```

## Performance

The optimized format provides significant improvements:

- **Loading Speed**: 10-100x faster than loading full DataFrame
- **Memory Usage**: Reduced by storing embeddings separately
- **Search Speed**: < 100ms for 750k+ items (with Faiss)

## License

MIT License