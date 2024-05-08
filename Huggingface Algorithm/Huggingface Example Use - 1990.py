import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score

# Initialize the Sentence Transformer model with a MiniLM architecture
model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths for tariff databases
file_2023 = 'tariff database_202305.xlsx'
file_1990 = '1990 WIP AZ.xlsx'

# Load the data from the specified Excel files
df_2023 = pd.read_excel(file_2023)
df_1990 = pd.read_excel(file_1990)

def find_similar_hs_codes_transformers(df_1990, df_2023, top_n=1):
    """
    Computes the top N similar HS codes for each product description from 1990 data.
    
    Args:
        df_1990 (DataFrame): DataFrame containing 1990 product data.
        df_2023 (DataFrame): DataFrame containing 2023 HS codes and descriptions.
        top_n (int): Number of top matches to return.
        
    Returns:
        list: A list of tuples containing matched HS codes and similarity scores.
    """
    # Combine product descriptions from 1990, handling missing values
    df_1990['combined_description'] = df_1990['ProductDescription'].fillna('')

    # Generate embeddings for descriptions from both datasets
    embeddings_1990 = model.encode(df_1990['combined_description'].tolist(), convert_to_tensor=True)
    embeddings_2023 = model.encode(df_2023['brief_description'].tolist(), convert_to_tensor=True)

    # Calculate cosine similarity scores
    cosine_scores = util.pytorch_cos_sim(embeddings_1990, embeddings_2023)

    # Extract top N matches based on similarity scores
    hs_code_matches = []
    for i in range(len(df_1990)):
        top_results = np.argsort(-cosine_scores[i].cpu().numpy())[:top_n]
        matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[i][j].item()) for j in top_results]
        hs_code_matches.append(matched_hs_codes)

    return hs_code_matches

def calculate_f1_scores(y_true, y_pred, digit_level=10):
    """
    Calculates the F1 score for HS code predictions, truncated to various digit levels.
    
    Args:
        y_true (list): List of actual HS codes.
        y_pred (list): List of predicted HS codes.
        digit_level (int): The number of digits to consider for the HS code comparison.
        
    Returns:
        float: The weighted F1 score.
    """
    y_true_truncated = [str(code)[:digit_level] for code in y_true]
    y_pred_truncated = [str(code)[:digit_level] for code in y_pred]

    return f1_score(y_true_truncated, y_pred_truncated, average='weighted')

def match_and_export_hs_codes_transformers(df_1990, df_2023, output_file_path):
    """
    Matches HS codes and exports the results along with F1 scores to a CSV file.
    
    Args:
        df_1990 (DataFrame): DataFrame containing 1990 product data.
        df_2023 (DataFrame): DataFrame containing 2023 HS codes and descriptions.
        output_file_path (str): Path to save the output CSV file.
        
    Returns:
        DataFrame: The exported DataFrame containing matched results and F1 scores.
    """
    all_matches = find_similar_hs_codes_transformers(df_1990, df_2023)

    export_data = []
    for i, matches in enumerate(all_matches):
        for hs_code, score in matches:
            df_2023_row = df_2023[df_2023['hts8'] == hs_code].iloc[0]
            original_hs_code = df_1990.iloc[i]['ProductCode']  # Assume the correct column name for actual HS code
            f1_hs10 = calculate_f1_scores([original_hs_code], [hs_code], 10)
            f1_hs6 = calculate_f1_scores([original_hs_code], [hs_code], 6)
            f1_hs4 = calculate_f1_scores([original_hs_code], [hs_code], 4)

            export_data.append({
                '1990 Product': df_1990.iloc[i]['ProductDescription'],
                'Original HS Code': original_hs_code,
                'Matched HS Code': hs_code,
                '2023 Description': df_2023_row['brief_description'],
                'F1 Score HS10': f1_hs10,
                'F1 Score HS6': f1_hs6,
                'F1 Score HS4': f1_hs4,
                'Similarity Score': score
            })

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)

    return export_df

# Path for the output CSV file and executing the matching and export process
output_csv_path = 'matched_hs_codes_1990_to_2023_transformers.csv'
exported_df_transformers = match_and_export_hs_codes_transformers(df_1990, df_2023, output_csv_path)
print(exported_df_transformers.head())
