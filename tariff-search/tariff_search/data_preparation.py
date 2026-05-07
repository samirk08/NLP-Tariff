"""
Data preparation utilities to convert the large DataFrame pickle 
into separate efficient files.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def prepare_data_files(
    input_pickle_path: str,
    output_dir: str,
    embedding_column: str = 'description_embedding',
    columns_to_keep: list = None
):
    """
    Convert a large DataFrame with embeddings into separate efficient files.
    
    This function:
    1. Loads the DataFrame with embeddings
    2. Extracts embeddings into a numpy array
    3. Saves metadata separately without embeddings
    4. Creates efficient storage format
    
    Args:
        input_pickle_path: Path to the input pickle file with embeddings
        output_dir: Directory to save the prepared files
        embedding_column: Name of the column containing embeddings
        columns_to_keep: List of columns to keep in metadata (None = keep all except embeddings)
    
    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_pickle_path}")
    df = pd.read_pickle(input_pickle_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings_list = df[embedding_column].tolist()
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # Save embeddings
    embeddings_path = output_dir / 'embeddings.npy'
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings_array)
    
    # Prepare metadata (without embeddings)
    if columns_to_keep is None:
        # Keep all columns except embeddings
        columns_to_keep = [col for col in df.columns if col != embedding_column]
    
    metadata = df[columns_to_keep].copy()
    
    # Save metadata
    metadata_path = output_dir / 'metadata.pkl'
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create info file
    info = {
        'n_items': len(df),
        'embedding_dim': embeddings_array.shape[1],
        'columns': list(metadata.columns),
        'embedding_column': embedding_column
    }
    
    info_path = output_dir / 'info.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(info, f)
    
    logger.info("Data preparation complete!")
    logger.info(f"Embeddings shape: {embeddings_array.shape}")
    logger.info(f"Metadata shape: {metadata.shape}")
    
    # Print file sizes
    embeddings_size = embeddings_path.stat().st_size / (1024 * 1024)
    metadata_size = metadata_path.stat().st_size / (1024 * 1024)
    logger.info(f"Embeddings file size: {embeddings_size:.2f} MB")
    logger.info(f"Metadata file size: {metadata_size:.2f} MB")


def verify_prepared_data(data_dir: str):
    """
    Verify that prepared data files are valid and can be loaded.
    
    Args:
        data_dir: Directory containing prepared files
    """
    data_dir = Path(data_dir)
    
    # Check files exist
    required_files = ['embeddings.npy', 'metadata.pkl', 'info.pkl']
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required file {file_name} not found in {data_dir}")
    
    # Load and verify
    logger.info("Verifying prepared data...")
    
    # Load info
    with open(data_dir / 'info.pkl', 'rb') as f:
        info = pickle.load(f)
    
    # Load embeddings
    embeddings = np.load(data_dir / 'embeddings.npy')
    assert embeddings.shape[0] == info['n_items'], "Embeddings count mismatch"
    assert embeddings.shape[1] == info['embedding_dim'], "Embedding dimension mismatch"
    
    # Load metadata
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    assert len(metadata) == info['n_items'], "Metadata count mismatch"
    
    logger.info(f"✓ Verified {info['n_items']} items with {info['embedding_dim']}-dim embeddings")
    logger.info(f"✓ Metadata columns: {', '.join(info['columns'][:5])}...")
    
    return True


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare tariff search data")
    parser.add_argument("input_pickle", help="Path to input pickle file with embeddings")
    parser.add_argument("output_dir", help="Directory to save prepared files")
    parser.add_argument("--embedding-column", default="description_embedding", 
                       help="Name of embedding column")
    parser.add_argument("--verify", action="store_true", help="Verify after preparation")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    prepare_data_files(
        args.input_pickle,
        args.output_dir,
        embedding_column=args.embedding_column
    )
    
    if args.verify:
        verify_prepared_data(args.output_dir)


if __name__ == "__main__":
    main()