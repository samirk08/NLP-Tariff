import pandas as pd
import numpy as np
import os
import logging
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from functools import partial
from fuzzywuzzy import fuzz

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the SentenceTransformer model for embedding text
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup API key for OpenAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = "your-api-key"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# File paths for the data
file_2023 = 'tariff database_202305.xlsx'
input_file = 'your_input_file.xlsx'  # Replace with your actual input file

# Load the data
df_2023 = pd.read_excel(file_2023)
df_input = pd.read_excel(input_file)

# Cache for storing enhanced descriptions to avoid repeated API calls
description_cache = {}

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(50))
def ask_gpt(prompt, system_prompt, model_name="gpt-4"):
    """
    Queries GPT model to enhance descriptions or generate content based on prompts.
    
    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str): Instructions for the model on how to handle the user prompt.
        model_name (str): Specifies the model version.
    
    Returns:
        str: The model's response text.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def find_most_similar_hs_code(description, df_2023, top_n=1):
    """
    Finds the most similar HS code in the 2023 dataset for a given description.
    
    Args:
        description (str): Description to match.
        df_2023 (DataFrame): DataFrame with HS codes and descriptions.
        top_n (int): Number of top results to return.
    
    Returns:
        tuple: The top matching HS code and its similarity score.
    """
    description_embedding = model.encode(description, convert_to_tensor=True)
    embeddings_2023 = model.encode(df_2023['brief_description'].tolist(), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings_2023)

    top_results = np.argsort(-cosine_scores.cpu().numpy())[0][:top_n]
    matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[0][j].item()) for j in top_results]

    return matched_hs_codes[0] if matched_hs_codes else ('', 0)

def calculate_similarity_with_actual_hs_code(predicted_hs_code, actual_hs_code):
    """
    Calculates the similarity score between predicted and actual HS codes.
    
    Args:
        predicted_hs_code (str): Predicted HS code.
        actual_hs_code (str): Actual HS code.
    
    Returns:
        float: Similarity score normalized between 0 and 1.
    """
    similarity = fuzz.ratio(str(predicted_hs_code), str(actual_hs_code))
    return similarity / 100.0

def process_item(row, df_2023, df_manual_coding):
    """
    Processes each item to enhance descriptions, find HS codes, and calculate similarity scores.
    
    Args:
        row (Pandas Series): A row from the DataFrame.
        df_2023 (DataFrame): DataFrame with 2023 HS code descriptions.
        df_manual_coding (DataFrame): DataFrame with manual coding for comparison.
    
    Returns:
        dict: A dictionary with processed item details including HS codes and similarity score.
    """
    item_description = f"{row.item} {row.description_1}".strip() if pd.notna(row.item) and pd.notna(row.description_1) else ''
    
    if item_description and item_description not in description_cache:
        system_prompt = "Enhance the description for better HS code matching."
        enhanced_description = ask_gpt(item_description, system_prompt)
        description_cache[item_description] = enhanced_description
    
    enhanced_description = description_cache.get(item_description, '')
    if enhanced_description:
        custom_hs_code_prompt = f"Assuming you have access to a complete 2023 HS code database, generate a plausible HS code and provide a detailed description for the following 1990 product description: {enhanced_description}"
        enhanced_description_gpt = ask_gpt(custom_hs_code_prompt, system_prompt)
        closest_hs_code_gpt, _ = find_most_similar_hs_code(enhanced_description_gpt, df_2023)

        # Retrieve actual HS code from manual coding for comparison
        matched_rows = df_manual_coding[df_manual_coding['brief_description'].str.lower() == row.item.lower()]
        actual_hs_code = matched_rows['hs'].iloc[0] if not matched_rows.empty else 'No Match'

        similarity_score = calculate_similarity_with_actual_hs_code(closest_hs_code_gpt, actual_hs_code)

        return {
            'Input Item': item_description,
            'Predicted HS Code': closest_hs_code_gpt,
            'Actual HS Code': actual_hs_code,
            'Similarity Score': similarity_score
        }
    else:
        logging.warning(f"Failed to enhance description: {item_description}")
    return None

def match_and_export_hs_codes_gpt(df_1789, df_2023, df_manual_coding, output_file_path):
    """
    Matches HS codes and exports the results to a CSV file.
    
    Args:
        df_1789 (DataFrame): DataFrame with initial data.
        df_2023 (DataFrame): DataFrame with 2023 HS codes.
        df_manual_coding (DataFrame): DataFrame for manual HS code verification.
        output_file_path (str): Path to the output CSV file.
    
    Returns:
        DataFrame: The exported results as a DataFrame.
    """
    export_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        process_func = partial(process_item, df_2023=df_2023, df_manual_coding=df_manual_coding)
        results = executor.map(process_func, df_1789.itertuples(index=False))
        export_data.extend(result for result in results if result)

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)
    return export_df

# Execution part
output_csv_path = 'gpt4-try1.csv'
exported_df_gpt = match_and_export_hs_codes_gpt(df_input, df_2023, df_manual_coding, output_csv_path)
print(exported_df_gpt.head())
