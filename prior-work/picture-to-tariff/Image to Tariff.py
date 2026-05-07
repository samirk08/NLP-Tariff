import pandas as pd
import numpy as np
import os
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sentence_transformers import SentenceTransformer, util
from functools import partial
from fuzzywuzzy import fuzz
import torch
import base64
import requests

# Configure the device to use for computation (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the SentenceTransformer model and move it to the configured device
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Disable parallelism in tokenizers to avoid threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set your OpenAI API key here for security
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the OpenAI client
client = openai.OpenAI()

# Function to encode images to base64 for sending to GPT
def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image path for analysis
image_path = "/path/to/your/image.jpg"
base64_image = encode_image(image_path)

# Headers for making HTTP requests to OpenAI
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Prepare the payload for the OpenAI API call
payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please provide a concise description of the primary item in this image."
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ],
    "max_tokens": 300
}

# Making the API request to OpenAI
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_data = response.json()

# Extract the response from the API call
user_input = response_data['choices'][0]['message']['content']
print(user_input)

# Load the tariff data from an Excel file
df_2023 = pd.read_excel("/path/to/tariff/database_202305.xlsx")
brief_descriptions = df_2023['brief_description'].tolist()

# Load pre-computed embeddings from a file and move to the configured device
embeddings_2023 = torch.load("/path/to/embeddings.pt").to(device)

# Decorated function to use GPT for enhancing descriptions with retries
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(50))
def ask_gpt(prompt, system_prompt, model_name="gpt-4"):
    """Requests GPT to enhance a description based on a prompt."""
    response = client.chat.completions.create(model=model_name,
                                              messages=[
                                                  {"role": "system", "content": system_prompt},
                                                  {"role": "user", "content": prompt}
                                              ],
                                              max_tokens=300,
                                              temperature=0.0)
    return response.choices[0].message.content.strip()

# Function to calculate similarity between a description and the database
def calculate_similarity(description, embeddings_2023, df_2023):
    """Calculates similarity of a description with HS codes using embeddings."""
    description_embedding = model.encode(description, convert_to_tensor=True).to(device)
    cosine_scores = util.pytorch_cos_sim(description_embedding, embeddings_2023)
    
    top_result = torch.argmax(cosine_scores, dim=1)
    matched_hs_code = df_2023.iloc[top_result.item()]['hts8']
    similarity_score = cosine_scores[0, top_result.item()].item()
    matched_description = df_2023.iloc[top_result.item()]['brief_description']

    return matched_hs_code, similarity_score, matched_description

# Main function to process user input and compare HS code predictions
def process_and_compare(user_input):
    """Processes user input, enhances it with GPT, and calculates HS code similarity."""
    system_prompt = "Enhance this product description for tariff classification:"
    enhanced_description_gpt = ask_gpt(user_input, system_prompt)
    
    gpt_hs_code, gpt_similarity_score, gpt_matched_description = calculate_similarity(enhanced_description_gpt, embeddings_2023, df_2023)
    hf_hs_code, hf_similarity_score, hf_matched_description = calculate_similarity(user_input, embeddings_2023, df_2023)
    
    # Select the best result based on similarity scores
    if gpt_similarity_score > hf_similarity_score:
        chosen_hs_code = gpt_hs_code
        final_similarity_score = gpt_similarity_score
        method_used = 'GPT'
        chosen_description = gpt_matched_description
    else:
        chosen_hs_code = hf_hs_code
        final_similarity_score = hf_similarity_score
        method_used = 'HF'
        chosen_description = hf_matched_description
    
    return chosen_hs_code, final_similarity_score, method_used, chosen_description

# Example of processing input and getting results
chosen_hs_code, final_similarity_score, method_used, chosen_description = process_and_compare(user_input)
print(f"Method Used: {method_used}")
print(f"Matched HS Code: {chosen_hs_code}, Similarity Score: {final_similarity_score}")
print(f"Matched Description: {chosen_description}")
