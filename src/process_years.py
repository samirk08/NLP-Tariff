import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from rapidfuzz import fuzz
import pickle

# Set up the model and tokenizer
def setup_model_and_tokenizer():
    token = os.environ.get("HF_TOKEN")  # optional; the public model loads without auth
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

### Similarity Functions ###

def normalized_levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Calculate the normalized Levenshtein similarity between two strings.
    """
    if not text1 or not text2:
        return 0.0
    return fuzz.ratio(text1.strip().lower(), text2.strip().lower()) / 100.0

def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity between two embeddings.
    """
    if embedding1.dim() > 1:
        embedding1 = embedding1.squeeze(0)
    if embedding2.dim() > 1:
        embedding2 = embedding2.squeeze(0)
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def jaccard_similarity(s1:set, s2:set)->float:
    return len(s1 & s2)/len(s1 | s2) if s1 and s2 else 0.0


### Process tarrifs function ###
def process_tariffs(
    description_N1: str, 
    hs_code_N1: str, 
    embedding_N1: torch.Tensor,
    descriptions_N: list[str], 
    hs_codes_N: list[str], 
    embeddings_N: torch.Tensor, 
    token_cache_N: list[set[str]],
    alpha: float = 0.7, 
    beta: float = 0.2, 
    gamma: float = 0.1, 
    top_k: int = 500
) -> dict:
    """
    Find the most similar description and HS code for a given description and HS code.

    Parameters:
    - description_N1 (str): The description to map.
    - hs_code_N1 (str): The HS code corresponding to description_N1.
    - embedding_N1 (torch.Tensor): The embedding of description_N1.
    - descriptions_N (list[str]): List of descriptions to compare against.
    - hs_codes_N (list[str]): List of HS codes corresponding to descriptions_N.
    - embeddings_N (torch.Tensor): Tensor of embeddings for descriptions_N.
    - token_cache_N (list[set[str]]): Precomputed token sets for descriptions_N.
    - alpha (float): Weight for cosine similarity (default: 0.7).
    - beta (float): Weight for Jaccard similarity (default: 0.2).
    - gamma (float): Weight for Levenshtein similarity (default: 0.1).
    - top_k (int): Number of top candidates to consider (default: 500).

    Returns:
    - dict: A dictionary containing the best match details, including similarities and mapped values.
    """
    """Want to find the most similar description and hs_code for each description_N1 and hs_code_N1"""

    ## Find the most similar descriptions for each description_N1
    embedding_N1 = embedding_N1.unsqueeze(0) # Add batch dimension
    # Calculate similarities
    cosine_scores = F.cosine_similarity(embedding_N1, embeddings_N, dim=1) 
    cosine_scores_np = cosine_scores.cpu().numpy()
    # Get the top k indices
    top_k_indices = np.argsort(cosine_scores_np)[-top_k:][::-1]

    best_similarity = -1
    best_candidate_idx = -1
    best_cos = best_jacc = best_lev = 0
    tokens_N1 = set(description_N1.lower().split())

    for idx in top_k_indices:
        candidate_description = descriptions_N[idx]
        candidate_cos = cosine_scores_np[idx]
        candidate_tokens = token_cache_N[idx]
        candidate_jacc = jaccard_similarity(tokens_N1, candidate_tokens)
        candidate_lev = normalized_levenshtein_similarity(description_N1, candidate_description)
        # similarity = a * Cos + b * Jaccard + c * Levenshtein
        similarity = alpha * candidate_cos + beta * candidate_jacc + gamma * candidate_lev

        if similarity > best_similarity:
            best_similarity = similarity
            best_candidate_idx = idx
            best_cos = candidate_cos
            best_jacc = candidate_jacc
            best_lev = candidate_lev
        
    best_description = descriptions_N[best_candidate_idx]
    best_hs_code = hs_codes_N[best_candidate_idx]

    return {
        "HS_N1": hs_code_N1,
        "Description_N1": description_N1,
        "Mapped_HS": best_hs_code,
        "Mapped_Description": best_description,
        "Cosine_Similarity": best_cos,
        "Jaccard_Similarity": best_jacc,
        "Levenshtein_Similarity": best_lev,
        "Combined_Similarity": best_similarity
    }

def process_tariffs_parallel(df_N1, df_N, tokenizer, model, device, alpha=0.7, beta=0.2, gamma=0.1, top_k=500):
    """
    Process tariffs in parallel using ThreadPoolExecutor.
    """
    max_workers = min(32, os.cpu_count() + 4)

    # drop na descriptions
    df_N1 = df_N1.dropna(subset=['brief_description']).copy()
    df_N = df_N.dropna(subset=['brief_description']).copy()
    df_N1['brief_description'] = df_N1['brief_description'].astype(str)
    df_N['brief_description'] = df_N['brief_description'].astype(str)

    # convert descriptions and hs to list
    descriptions_N1 = df_N1['brief_description'].tolist()
    hs_codes_N1 = df_N1['hs'].tolist()
    descriptions_N = df_N['brief_description'].tolist()
    hs_codes_N = df_N['hs'].tolist()

    # get the years
    years_N1 = df_N1['year'].iloc[0]
    years_N = df_N['year'].iloc[0]

    # open embeddings pickle files for these years
    with open(f"embeddings/embeddings_{years_N1}.pkl", "rb") as f:
        embeddings_N1 = pickle.load(f)
    with open(f"embeddings/embeddings_{years_N}.pkl", "rb") as f:
        embeddings_N = pickle.load(f)

    # ensure embeddings are torch tensors on correct device
    embeddings_N1 = torch.as_tensor(embeddings_N1, device=device, dtype=torch.float32)
    embeddings_N = torch.as_tensor(embeddings_N, device=device, dtype=torch.float32)

    # create token cache
    token_cache_N = [set(desc.lower().split()) for desc in descriptions_N]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_tariffs, desc_N1, hs_N1, emb_N1,
                descriptions_N, hs_codes_N, embeddings_N, token_cache_N,
                alpha, beta, gamma, top_k
            )
            for idx, (desc_N1, hs_N1, emb_N1) in enumerate(zip(descriptions_N1, hs_codes_N1, embeddings_N1))
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {years_N1} to {years_N}"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing item: {e}")
    
    mapped_df = pd.DataFrame(results)
    return mapped_df


def main():
    # Initialize model and tokenizer (currently unused, but kept for future use)
    tokenizer, model, device = setup_model_and_tokenizer()
    all_tariffs = pd.read_csv('./all_tariffs.csv')
    df = all_tariffs[['year', 'hs', 'brief_description']]

    # clean the data
    df = df.dropna(subset=['brief_description']).copy()
    df['brief_description'] = df['brief_description'].astype(str)

    # get a list of years
    years = df['year'].unique()
    years = sorted(years)

    # process all adjacent years
    for i in range(len(years) - 1):
        year_N = years[i]
        year_N1 = years[i + 1]
        
        output_filename = f'mapped_{year_N1}_{year_N}.csv'
        if os.path.exists(output_filename):
            print(f"Output file {output_filename} already exists. Skipping...")
            continue
    
        print(f"Processing {year_N1} to {year_N}...")

        # filter the dataframes for the current year
        df_N = df[df['year'] == year_N].copy()
        df_N1 = df[df['year'] == year_N1].copy()

        # check for empty dataframes
        if df_N.empty or df_N1.empty:
            print(f"One of the dataframes for {year_N} or {year_N1} is empty. Skipping...")
            continue

        mapped_df = process_tariffs_parallel(df_N1, df_N, tokenizer, model, device)

        try:
            mapped_df.to_csv(output_filename, index=False)
            print(f"Saved mapped data to {output_filename}")
        except Exception as e:
            print(f"Error saving mapped data to {output_filename}: {e}")

    print("All years processed.")
        
if __name__ == "__main__": 
    main()