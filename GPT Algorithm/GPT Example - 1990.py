import pandas as pd
import os
import openai
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy.spatial.distance import cosine
import numpy as np
from functools import partial

# Initialize logging for debugging and tracking
logging.basicConfig(level=logging.INFO)

# Set your OpenAI API key as an environment variable for security
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = "your-api-key"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
client = openai.OpenAI()

# File paths to the datasets
file_2023 = '/path/to/tariff/database_202305.xlsx'
file_1990 = '/path/to/1990/database.xlsx'

# Load the data from Excel files
df_2023 = pd.read_excel(file_2023)
df_1990 = pd.read_excel(file_1990)

def get_text_embedding(text):
    """
    Fetches the embedding for a given text using OpenAI's embedding model.
    
    Args:
        text (str): Text to embed.
    
    Returns:
        ndarray: The embedding vector of the provided text.
    """
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def find_closest_hs_code(description, hs_code_df):
    """
    Identifies the closest HS code from the provided dataframe based on cosine similarity.
    
    Args:
        description (str): Description of the product.
        hs_code_df (DataFrame): DataFrame containing HS codes and their descriptions.
    
    Returns:
        tuple: The closest HS code, its description, and the similarity score.
    """
    description_embedding = get_text_embedding(description)

    # Function to calculate cosine similarity for each row
    def calculate_similarity(row):
        hs_code_embedding = get_text_embedding(row['brief_description'])
        return 1 - cosine(description_embedding, hs_code_embedding)

    # Apply similarity calculation and find the best match
    hs_code_df['similarity_score'] = hs_code_df.apply(calculate_similarity, axis=1)
    best_match_row = hs_code_df.loc[hs_code_df['similarity_score'].idxmax()]

    return best_match_row['hts8'], best_match_row['brief_description'], best_match_row['similarity_score']

def process_item(row, df_2023):
    """
    Processes each item to find the closest HS code.
    
    Args:
        row (Series): A row from the DataFrame.
        df_2023 (DataFrame): DataFrame with 2023 HS codes and descriptions.
    
    Returns:
        dict: Processed item details including matched HS code and similarity score.
    """
    description_to_process = row['ProductDescription'] if pd.notna(row['ProductDescription']) else ''
    if description_to_process.strip():
        closest_hs_code, product_description, similarity_score = find_closest_hs_code(description_to_process, df_2023)
        return {
            '1990 Product Code': row['ProductCode'],
            '1990 Description': row['ProductDescription'],
            'Matched HS Code': closest_hs_code,
            '2023 Product Description': product_description,
            'Similarity Score': similarity_score
        }
    else:
        logging.warning(f"Empty or invalid description for item index: {row.name}")

def match_and_export_hs_codes(df_1990, df_2023, output_file_path):
    """
    Matches HS codes for all items and exports the results to a CSV file.
    
    Args:
        df_1990 (DataFrame): DataFrame with 1990 product data.
        df_2023 (DataFrame): DataFrame with 2023 HS code descriptions.
        output_file_path (str): Path to save the output CSV.
    
    Returns:
        DataFrame: The exported results as a DataFrame.
    """
    export_data = []
    process_item_with_df_2023 = partial(process_item, df_2023=df_2023)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_item_with_df_2023, [row for _, row in df_1990.iterrows()])
        export_data.extend(result for result in results if result)

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)
    return export_df

# Define the output path for the matched HS codes CSV and execute the function
output_csv_path = 'HS_Code_Match_Output.csv'
exported_df = match_and_export_hs_codes(df_1990, df_2023, output_csv_path)
print(exported_df.head())
