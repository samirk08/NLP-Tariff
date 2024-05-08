import pandas as pd
import numpy as np
import os
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor
import logging
from sentence_transformers import SentenceTransformer, util
from functools import partial
from fuzzywuzzy import fuzz
# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your OpenAI API key as an environment variable for security
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# File paths
file_2023 = 'tariff database_202305.xlsx'
input_file = ''

# Load the data
df_2023 = pd.read_excel(file_2023)
df_input = pd.read_excel(input_file)

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
def find_most_similar_hs_code(description, df_2023, top_n=1):
    description_embedding = model.encode(description, convert_to_tensor=True)
    embeddings_2023 = model.encode(df_2023['brief_description'].tolist(), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings_2023)

    top_results = np.argsort(-cosine_scores.cpu().numpy())[0][:top_n]
    matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[0][j].item()) for j in top_results]

    return matched_hs_codes[0] if matched_hs_codes else ('', 0)
def calculate_similarity_with_actual_hs_code(predicted_hs_code, actual_hs_code):
    predicted_hs_code_str = str(predicted_hs_code)
    actual_hs_code_str = str(actual_hs_code)
    similarity = fuzz.ratio(predicted_hs_code_str, actual_hs_code_str)
    return similarity / 100.0
def process_item(row, df_2023, df_manual_coding):
    item_description = row.item if pd.notna(row.item) else ''
    description_1 = row.description_1 if pd.notna(row.description_1) else ''
    description_to_enhance = f"{item_description} {description_1}".strip()

    if description_to_enhance:
        if description_to_enhance in description_cache:
            enhanced_description = description_cache[description_to_enhance]
        else:
            system_prompt = "Enhance the description for better HS code matching."
            enhanced_description = ask_gpt(description_to_enhance, system_prompt)
            description_cache[description_to_enhance] = enhanced_description

        if enhanced_description:
            custom_hs_code_prompt = "Assuming you have access to a complete 2023 HS code database, generate a plausible HS code and provide a detailed description for the following 1990 product description as if you are retrieving it from the database. Please format your response as an HS code followed by its description, without disclaimers or statements about browsing capabilities: " + enhanced_description
            enhanced_description_gpt = ask_gpt(custom_hs_code_prompt, system_prompt)
            closest_hs_code_gpt, _ = find_most_similar_hs_code(enhanced_description_gpt, df_2023)

            # Retrieve the actual HS code from manual coding
            matched_rows = df_manual_coding[df_manual_coding['brief_description'].str.lower() == row.item.lower()]
            if not matched_rows.empty:
                actual_hs_code = matched_rows['hs'].iloc[0]
            else:
                actual_hs_code = None  # or some default value such as 'No Match'

            # Calculate the similarity score using fuzzywuzzy
            similarity_score = calculate_similarity_with_actual_hs_code(closest_hs_code_gpt, actual_hs_code) if actual_hs_code else None

            return {
                'Input Item': description_to_enhance,
                'Predicted HS Code': closest_hs_code_gpt,
                'Actual HS Code': actual_hs_code,
                'Similarity Score': similarity_score
            }
        else:
            logging.warning(f"Failed to enhance description: {description_to_enhance}")
    return None
def match_and_export_hs_codes_gpt(df_1789, df_2023, df_manual_coding, output_file_path):
    export_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Adjust the partial function call to pass df_manual_coding as well
        process_func = partial(process_item, df_2023=df_2023, df_manual_coding=df_manual_coding)
        results = list(executor.map(process_func, df_1789.itertuples(index=False)))
        export_data.extend([result for result in results if result])

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)
    return export_df
# Now, you also need to pass df_manual_coding when calling match_and_export_hs_codes_gpt
output_csv_path = 'gpt4-try1.csv'
exported_df_gpt = match_and_export_hs_codes_gpt(df_input, df_2023, df_manual_coding, output_csv_path)
print(exported_df_gpt.head())