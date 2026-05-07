"""
Example usage of the tariff_search package.

This script demonstrates how to:
1. Prepare data from existing pickle file
2. Search for similar tariff descriptions
3. Display and analyze results
"""

import pandas as pd
from tariff_search import TariffSearch, prepare_data_files
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Prepare data (only needs to be done once)
    print("Step 1: Preparing data files...")
    print("This converts the large pickle file into efficient format.")
    
    # Uncomment to prepare data from your pickle file
    # prepare_data_files(
    #     input_pickle_path="df_with_embeddings.pkl",
    #     output_dir="tariff_search/data",
    #     embedding_column="description_embedding"
    # )
    
    # Step 2: Initialize search engine
    print("\nStep 2: Initializing search engine...")
    searcher = TariffSearch(data_dir="tariff_search/data")
    
    # Step 3: Example searches
    print("\nStep 3: Performing example searches...")
    
    example_queries = [
        "Cotton cloth, bleached, containing synthetic fiber",
        "Leather shoes for men",
        "Electronic calculators",
        "Fresh apples",
        "Wooden furniture"
    ]
    
    for query in example_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Search
        results = searcher.search(query, top_k=3)
        
        # Display results
        for idx, row in results.iterrows():
            print(f"\nRank {idx + 1}:")
            print(f"  Combined Similarity: {row['combined_similarity']:.3f}")
            print(f"  Cosine Similarity: {row['cosine_similarity']:.3f}")
            print(f"  Description: {row.get('Description_N', 'N/A')}")
            print(f"  Historical Code: {row.get('HS_N', 'N/A')}")
            print(f"  Year: {row.get('year_N', 'N/A')}")
            if 'best_match_hscode' in row:
                print(f"  Modern HS Code: {row['best_match_hscode']}")
                print(f"  Modern Description: {row.get('best_match_description', 'N/A')[:80]}...")
    
    # Step 4: Batch search example
    print(f"\n\n{'='*60}")
    print("Batch Search Example")
    print('='*60)
    
    batch_queries = [
        "Steel pipes",
        "Plastic containers",
        "Coffee beans"
    ]
    
    batch_results = searcher.batch_search(batch_queries, top_k=2)
    
    for query, results in batch_results.items():
        print(f"\nQuery: {query}")
        if not results.empty:
            best_match = results.iloc[0]
            print(f"  Best match: {best_match.get('Description_N', 'N/A')} (similarity: {best_match['combined_similarity']:.3f})")


def interactive_search():
    """Interactive search mode."""
    print("\nTariff Search Interactive Mode")
    print("Type 'quit' to exit\n")
    
    # Initialize searcher
    searcher = TariffSearch(data_dir="tariff_search/data")
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        # Search
        results = searcher.search(query, top_k=5)
        
        # Display results
        print(f"\nFound {len(results)} results:")
        for idx, row in results.iterrows():
            print(f"\n{idx + 1}. Similarity: {row['combined_similarity']:.3f}")
            print(f"   {row.get('Description_N', 'N/A')}")
            print(f"   Code: {row.get('HS_N', 'N/A')} (Year: {row.get('year_N', 'N/A')})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        main()