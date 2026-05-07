"""
Main search functionality for tariff descriptions.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pickle
import os
from typing import Optional, List, Tuple, Dict
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)

class TariffSearch:
    """
    Efficient tariff description search using precomputed embeddings.
    
    This class loads metadata and embeddings separately for fast searching.
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 use_faiss: bool = True,
                 device: str = None):
        """
        Initialize the TariffSearch engine.
        
        Args:
            data_dir: Directory containing the preprocessed data files
            model_name: Transformer model to use for query embeddings
            use_faiss: Whether to use Faiss for fast similarity search
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
        """
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = Path(data_dir)
        
        # Load model and tokenizer
        self._load_model()
        
        # Load data
        self._load_data()
        
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _load_data(self):
        """Load the metadata and embeddings."""
        logger.info("Loading data files...")
        
        # Load metadata (without embeddings)
        metadata_path = self.data_dir / 'metadata.pkl'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load embeddings as numpy array
        embeddings_path = self.data_dir / 'embeddings.npy'
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        
        self.embeddings = np.load(embeddings_path)
        
        # Initialize Faiss index if requested
        if self.use_faiss:
            try:
                import faiss
                self._init_faiss_index()
            except ImportError:
                logger.warning("Faiss not installed. Falling back to numpy search.")
                self.use_faiss = False
                self.index = None
        else:
            self.index = None
            
        logger.info(f"Loaded {len(self.metadata)} items")
        
    def _init_faiss_index(self):
        """Initialize Faiss index for fast similarity search."""
        import faiss
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        self.index.add(self.embeddings)
        
        logger.info("Faiss index initialized")
        
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into an embedding."""
        inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            query_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_embedding = query_embedding.detach().cpu().numpy().flatten()
            
        return query_embedding
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               alpha: float = 0.7,
               beta: float = 0.2,
               gamma: float = 0.1,
               rerank: bool = True) -> pd.DataFrame:
        """
        Search for similar tariff descriptions.
        
        Args:
            query: The query description
            top_k: Number of results to return
            alpha: Weight for cosine similarity
            beta: Weight for Jaccard similarity
            gamma: Weight for Levenshtein similarity
            rerank: Whether to rerank results using combined similarity
            
        Returns:
            DataFrame with top_k most similar items
        """
        # Encode query
        query_embedding = self._encode_query(query)
        
        # Get candidates using vector similarity
        if rerank:
            # Get more candidates for reranking
            n_candidates = min(top_k * 10, len(self.metadata))
        else:
            n_candidates = top_k
            
        if self.use_faiss and self.index is not None:
            # Use Faiss for search
            import faiss
            query_norm = query_embedding.copy()
            faiss.normalize_L2(query_norm.reshape(1, -1))
            distances, indices = self.index.search(query_norm.reshape(1, -1), n_candidates)
            similarities = distances[0]  # Already cosine similarities
            indices = indices[0]
        else:
            # Use numpy for search
            norm_query = np.linalg.norm(query_embedding)
            norm_all = np.linalg.norm(self.embeddings, axis=1)
            dot_products = np.dot(self.embeddings, query_embedding)
            similarities = dot_products / (norm_query * norm_all)
            indices = np.argsort(similarities)[-n_candidates:][::-1]
            similarities = similarities[indices]
            
        # Get candidate rows
        candidates = []
        query_tokens = set(query.lower().split())
        
        for idx, sim in zip(indices, similarities):
            if idx < 0 or idx >= len(self.metadata):  # Faiss might return -1 for empty slots
                continue
                
            row = self.metadata.iloc[idx].copy()
            description = row.get('Description_N', '')
            
            if rerank and description:
                # Calculate additional similarities
                desc_tokens = set(str(description).lower().split())
                jac_sim = len(query_tokens & desc_tokens) / len(query_tokens | desc_tokens) if desc_tokens else 0.0
                lev_sim = fuzz.ratio(query.lower(), str(description).lower()) / 100.0
                combined_sim = alpha * sim + beta * jac_sim + gamma * lev_sim
            else:
                jac_sim = 0.0
                lev_sim = 0.0
                combined_sim = sim
                
            candidates.append({
                'idx': idx,
                'cosine_similarity': float(sim),
                'jaccard_similarity': jac_sim,
                'levenshtein_similarity': lev_sim,
                'combined_similarity': combined_sim,
                **row.to_dict()
            })
            
        # Sort by combined similarity if reranking
        if rerank:
            candidates.sort(key=lambda x: x['combined_similarity'], reverse=True)
            
        # Return top_k results
        results = pd.DataFrame(candidates[:top_k])
        
        # Reorder columns for better display
        if not results.empty:
            similarity_cols = ['combined_similarity', 'cosine_similarity', 'jaccard_similarity', 'levenshtein_similarity']
            other_cols = [col for col in results.columns if col not in similarity_cols + ['idx']]
            results = results[similarity_cols + other_cols]
            
        return results
    
    def batch_search(self, queries: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of query descriptions
            **kwargs: Arguments passed to search()
            
        Returns:
            Dictionary mapping query to results DataFrame
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, **kwargs)
        return results