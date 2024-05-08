import pandas as pd
import numpy as np
import os
import openai
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed  
import logging
from sentence_transformers import SentenceTransformer, util
from functools import partial
from fuzzywuzzy import fuzz
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = ""  
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# File paths
file_2023 = '/home/samirk08/UROP_SPRING_2024/UROP IAP 2024/Original Databases/tariff database_202305.xlsx'  # Update this path
file_1996 = '/home/samirk08/UROP_SPRING_2024/1996/1996_1000.xlsx' 

# Load the data
df_2023 = pd.read_excel(file_2023)
df_1996 = pd.read_excel(file_1996)

description_cache = {}
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(50))
def ask_gpt(prompt, system_prompt, model_name="gpt-4"):
    response = client.chat.completions.create(model=model_name,
                                              messages=[
                                                  {"role": "system", "content": system_prompt},
                                                  {"role": "user", "content": prompt}
                                              ],
                                              max_tokens=300,
                                              temperature=0.0)
    return response.choices[0].message.content.strip()
def batch_encode_descriptions(descriptions, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False).to(device)
        all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0)

# Pre-compute embeddings for the 2023 dataset to avoid redundant computation
brief_descriptions = df_2023['brief_description'].tolist()
embeddings_2023 = batch_encode_descriptions(brief_descriptions, model)
def find_most_similar_hs_code(description, embeddings_2023, df_2023, top_n=1):
    description_embedding = model.encode([description], convert_to_tensor=True).to(device)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings_2023).cpu()

    top_results = np.argsort(-cosine_scores.numpy())[0][:top_n]
    matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[0][j].item()) for j in top_results]

    return matched_hs_codes[0] if matched_hs_codes else ('', 0)

def process_item(row, embeddings_2023, df_2023):
    item_description = row.ProductDescription if pd.notna(row.ProductDescription) else ''
    # description_1 = row['Description 2'] if pd.notna(row['Description 2']) else ''
    description_to_enhance = f"{item_description}".strip()

    if description_to_enhance in description_cache:
        enhanced_description = description_cache[description_to_enhance]
    else:
        system_prompt = "Enhance the description for better HS code matching."
        enhanced_description = ask_gpt(description_to_enhance, system_prompt)
        description_cache[description_to_enhance] = enhanced_description

    if enhanced_description:
        closest_hs_code_gpt, confidence_score = find_most_similar_hs_code(enhanced_description, embeddings_2023, df_2023)
        
        associated_2023_description = df_2023[df_2023['hts8'] == closest_hs_code_gpt]['brief_description'].iloc[0] if closest_hs_code_gpt else "No Description Found"

        return {
            '1996 Item': description_to_enhance,
            'Predicted HS Code': closest_hs_code_gpt,
            'Associated 2023 Description': associated_2023_description,
            'Confidence Score': confidence_score
        }
    else:
        logging.warning(f"Failed to enhance description: {description_to_enhance}")
    return None
def match_and_export_hs_codes_gpt(df_1996, embeddings_2023, df_2023, output_file_path):
    export_data = []
    with ThreadPoolExecutor(max_workers=50) as executor:  # Adjusted to 28 based on your system's capabilities
        futures = [executor.submit(process_item, row, embeddings_2023, df_2023) for index, row in df_1996.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if result:
                export_data.append(result)
    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)
output_csv_path = '1996_GPT_Samlpe.csv'
match_and_export_hs_codes_gpt(df_1996, embeddings_2023, df_2023, output_csv_path)