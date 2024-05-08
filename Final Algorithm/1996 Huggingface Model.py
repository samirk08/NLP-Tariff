import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from fuzzywuzzy import fuzz
import torch

# Load the SentenceTransformer model, ensuring it uses GPU if available
model = SentenceTransformer('all-MiniLM-L6-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define file paths
file_2023 = '/path/to/2023/tariff/database.xlsx'
file_1996 = '/path/to/1996/data.xlsx'

# Load the data from Excel files
df_2023 = pd.read_excel(file_2023)
df_1996 = pd.read_excel(file_1996)

def batch_encode_descriptions(model, descriptions, batch_size=32):
    """
    Batch encodes descriptions into embeddings using a specified model.
    
    Args:
        model (SentenceTransformer): The transformer model used for encoding.
        descriptions (list): List of descriptions to encode.
        batch_size (int): The number of descriptions processed per batch.
    
    Returns:
        torch.Tensor: Tensor containing the concatenated embeddings.
    """
    all_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True).to(device)
        all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0)

# Pre-compute embeddings for the 2023 dataset to enhance performance
brief_descriptions = df_2023['brief_description'].tolist()
embeddings_2023 = batch_encode_descriptions(model, brief_descriptions)

def find_most_similar_hs_code(description, embeddings_2023, df_2023, top_n=1):
    """
    Identifies the most similar HS code for a given description.
    
    Args:
        description (str): Product description.
        embeddings_2023 (torch.Tensor): Pre-computed embeddings for 2023 HS codes.
        df_2023 (pd.DataFrame): DataFrame containing HS codes and descriptions.
        top_n (int): Number of top results to return.
    
    Returns:
        tuple: The predicted HS code and its confidence score.
    """
    description_embedding = model.encode([description], convert_to_tensor=True).to(device)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings_2023)
    top_results = np.argsort(-cosine_scores.cpu().numpy())[0][:top_n]
    top_index = top_results[0]
    predicted_hs_code = df_2023.iloc[top_index]['hts8']
    confidence_score = cosine_scores[0][top_index].item()
    return predicted_hs_code, confidence_score

def calculate_similarity_with_actual_hs_code(predicted_hs_code, actual_hs_code):
    """
    Calculates the similarity score between predicted and actual HS codes using fuzzy matching.
    
    Args:
        predicted_hs_code (str): Predicted HS code.
        actual_hs_code (str): Actual HS code.
    
    Returns:
        float: Similarity score normalized to 0-1 scale.
    """
    return fuzz.ratio(str(predicted_hs_code), str(actual_hs_code)) / 100.0

def process_item_and_predict_hs_code(row, embeddings_2023, df_2023):
    """
    Processes each item to predict HS codes.
    
    Args:
        row (pd.Series): Data row containing product details.
        embeddings_2023 (torch.Tensor): Embeddings for HS codes.
        df_2023 (pd.DataFrame): HS code data.
    
    Returns:
        dict: Processed item details with predicted HS code and confidence score.
    """
    combined_description = row['ProductDescription'].strip() if pd.notna(row['ProductDescription']) else ''
    predicted_hs_code, confidence_score = find_most_similar_hs_code(combined_description, embeddings_2023, df_2023)
    return {
        '1996 Item': combined_description,
        'Predicted HS Code': predicted_hs_code,
        'Confidence Score': confidence_score
    }

def match_and_export_parallel(df_1996, embeddings_2023, df_2023, output_file_path):
    """
    Matches HS codes for all items and exports the results in parallel.
    
    Args:
        df_1996 (pd.DataFrame): Data from 1996.
        embeddings_2023 (torch.Tensor): Pre-computed embeddings for HS codes.
        df_2023 (pd.DataFrame): HS code data.
        output_file_path (str): Path to save the output CSV.
    """
    with ThreadPoolExecutor(max_workers=50) as executor:
        process_func = partial(process_item_and_predict_hs_code, embeddings_2023=embeddings_2023, df_2023=df_2023)
        futures = [executor.submit(process_func, row) for _, row in df_1996.iterrows()]
        export_data = [future.result() for future in as_completed(futures)]

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)

# Define the output file path
output_csv_path = 'path/to/output/HF_1996_Sample.csv'
match_and_export_parallel(df_1996, embeddings_2023, df_2023, output_csv_path)
