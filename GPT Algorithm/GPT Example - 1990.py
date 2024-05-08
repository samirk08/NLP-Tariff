import pandas as pd
import os
import openai
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy.spatial.distance import cosine
import numpy as np
from functools import partial
from openai import OpenAI

# Set your OpenAI API key as an environment variable for security
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# File paths
file_2023 = '/Users/samirkadariya/Desktop/School/UROP IAP 2024/Original Databases/tariff database_202305.xlsx'
file_1990 = '/Users/samirkadariya/Desktop/School/UROP IAP 2024/Original Databases/1990_CUT.xlsx'

# Load the data
df_2023 = pd.read_excel(file_2023)
df_1990 = pd.read_excel(file_1990)


def get_text_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def find_closest_hs_code(description, hs_code_df):
    # Get embedding for the description
    description_embedding = get_text_embedding(description)

    # Function to calculate cosine similarity
    def calculate_similarity(row):
        hs_code_embedding = get_text_embedding(row['brief_description'])
        return 1 - cosine(description_embedding, hs_code_embedding)

    # Calculate similarity scores for each HS code
    hs_code_df['similarity_score'] = hs_code_df.apply(calculate_similarity, axis=1)

    # Find the row with the highest similarity score
    best_match_row = hs_code_df.loc[hs_code_df['similarity_score'].idxmax()]

    # Return the best match details
    return best_match_row['hts8'], best_match_row['brief_description'], best_match_row['similarity_score']

def process_item(row, df_2023):
    description_to_process = row['ProductDescription'] if pd.notna(row['ProductDescription']) else ''
    if description_to_process.strip():
        # Find the closest HS code match in 2023 HS codes
        closest_hs_code, product_description, similarity_score = find_closest_hs_code(description_to_process, df_2023)

        # Return the results
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
    export_data = []
    process_item_with_df_2023 = partial(process_item, df_2023=df_2023)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_item_with_df_2023, [row for _, row in df_1990.iterrows()])
        for result in results:
            if result:
                export_data.append(result)

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)
    return export_df

output_csv_path = 'HS_Code_Match_Output.csv'
exported_df = match_and_export_hs_codes(df_1990, df_2023, output_csv_path)
print(exported_df.head())
