"""
Script to convert your existing df_with_embeddings.pkl to the optimized format.
Run this once to prepare your data for fast searching.
"""

from tariff_search import prepare_data_files
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Converting existing data to optimized format...")
    print("This may take a few minutes but only needs to be done once.")
    
    # Convert the existing pickle file
    # Update this path to point to your actual df_with_embeddings.pkl file
    prepare_data_files(
        input_pickle_path="../df_with_embeddings.pkl",  # Points to parent directory
        output_dir="tariff_search/data",
        embedding_column="description_embedding"
    )
    
    print("\nConversion complete!")
    print("You can now use the tariff search module for fast queries.")
    print("\nExample usage:")
    print("  from tariff_search import TariffSearch")
    print("  searcher = TariffSearch()")
    print('  results = searcher.search("Cotton cloth")')