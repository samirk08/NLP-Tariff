"""
Command-line interface for tariff search.
"""

import argparse
import logging
from pathlib import Path
from .searcher import TariffSearch
from .data_preparation import prepare_data_files, verify_prepared_data


def search_command(args):
    """Handle search command."""
    # Initialize searcher
    searcher = TariffSearch(data_dir=args.data_dir)
    
    # Perform search
    results = searcher.search(
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        rerank=not args.no_rerank
    )
    
    # Display results
    if args.format == 'simple':
        for idx, row in results.iterrows():
            print(f"\nSimilarity: {row['combined_similarity']:.3f}")
            print(f"Description: {row.get('Description_N', 'N/A')}")
            print(f"Code: {row.get('HS_N', 'N/A')}")
            print("-" * 50)
    elif args.format == 'csv':
        print(results.to_csv(index=False))
    elif args.format == 'json':
        print(results.to_json(orient='records', indent=2))


def prepare_command(args):
    """Handle data preparation command."""
    prepare_data_files(
        input_pickle_path=args.input,
        output_dir=args.output,
        embedding_column=args.embedding_column
    )
    
    if args.verify:
        verify_prepared_data(args.output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tariff Search - Find similar tariff descriptions"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar tariffs")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--data-dir", "-d",
        default="tariff_search/data",
        help="Directory containing prepared data"
    )
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for cosine similarity"
    )
    search_parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Weight for Jaccard similarity"
    )
    search_parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Weight for Levenshtein similarity"
    )
    search_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking with combined similarity"
    )
    search_parser.add_argument(
        "--format", "-f",
        choices=["simple", "csv", "json"],
        default="simple",
        help="Output format"
    )
    search_parser.set_defaults(func=search_command)
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare data files")
    prepare_parser.add_argument("input", help="Input pickle file path")
    prepare_parser.add_argument("output", help="Output directory")
    prepare_parser.add_argument(
        "--embedding-column",
        default="description_embedding",
        help="Name of embedding column"
    )
    prepare_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify prepared data"
    )
    prepare_parser.set_defaults(func=prepare_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()