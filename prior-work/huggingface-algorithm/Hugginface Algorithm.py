import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model with a predefined configuration
model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths for the tariff data
file_2023 = 'tariff database_202305.xlsx'
file_1789 = '1789.xlsx'

# Load the datasets
df_2023 = pd.read_excel(file_2023)
df_1789 = pd.read_excel(file_1789)

def find_similar_hs_codes_transformers(df_1789, df_2023, top_n=1):
    """
    Compute the most similar HS codes from the 2023 dataset for each entry in the 1789 dataset.

    Args:
    df_1789 (DataFrame): Data from 1789 containing items and descriptions.
    df_2023 (DataFrame): Modern HS codes with descriptions.
    top_n (int): Number of top matches to return.

    Returns:
    list of tuples: Each tuple contains HS codes and similarity scores.
    """
    # Combine item and description fields and handle missing values
    df_1789['combined_description'] = df_1789['item'].fillna('') + ' ' + df_1789['description_1'].fillna('')

    # Generate embeddings for descriptions from both datasets
    embeddings_1789 = model.encode(df_1789['combined_description'].tolist(), convert_to_tensor=True)
    embeddings_2023 = model.encode(df_2023['brief_description'].tolist(), convert_to_tensor=True)

    # Calculate cosine similarity between embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings_1789, embeddings_2023)

    # Identify the top N matches for each description
    hs_code_matches = []
    for i in range(len(df_1789)):
        top_results = np.argsort(-cosine_scores[i].cpu().numpy())[:top_n]
        matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[i][j].item()) for j in top_results]
        hs_code_matches.append(matched_hs_codes)

    return hs_code_matches

def match_and_export_hs_codes_transformers(df_1789, df_2023, output_file_path):
    """
    Matches HS codes and exports the results to a CSV file.

    Args:
    df_1789 (DataFrame): Data from 1789.
    df_2023 (DataFrame): Modern HS codes with descriptions.
    output_file_path (str): Path to save the output CSV file.

    Returns:
    DataFrame: The matched HS codes along with their descriptions and similarity scores.
    """
    all_matches = find_similar_hs_codes_transformers(df_1789, df_2023)

    export_data = []
    for i, matches in enumerate(all_matches):
        for hs_code, score in matches:
            df_2023_row = df_2023[df_2023['hts8'] == hs_code].iloc[0]
            export_data.append({
                '1789 Item': df_1789.iloc[i]['item'],
                '1789 Description': df_1789.iloc[i]['description_1'],
                'Matched HS Code': hs_code,
                '2023 Description': df_2023_row['brief_description'],
                'Similarity Score': score
            })

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)

    return export_df

# Specify the output CSV file path and execute the matching and export process
output_csv_path = 'matched_hs_codes_1789_to_2023_transformers.csv'
exported_df_transformers = match_and_export_hs_codes_transformers(df_1789, df_2023, output_csv_path)
print(exported_df_transformers.head())
