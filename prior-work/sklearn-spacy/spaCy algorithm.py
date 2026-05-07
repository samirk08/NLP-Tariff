import pandas as pd
import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')  # Make sure to use a model that has vectors

file_2023 = 'tariff database_202305.xlsx'
file_1789 = '1789.xlsx'

df_2023 = pd.read_excel(file_2023)
df_1789 = pd.read_excel(file_1789)
# Function to calculate similarity using spaCy
def find_similar_hs_codes_spacy(df_1789, df_2023, top_n=1):
    # Combining item and description fields for 1789 data
    df_1789['combined_description'] = df_1789['item'].fillna('') + ' ' + df_1789['description_1'].fillna('')

    # Create spaCy Docs for each combined description in both datasets
    docs_1789 = [nlp(text) for text in df_1789['combined_description']]
    docs_2023 = [nlp(text) for text in df_2023['brief_description']]

    # Calculate similarity scores
    similarity_scores = []
    for doc_1789 in docs_1789:
        scores = [doc_1789.similarity(doc_2023) for doc_2023 in docs_2023]
        similarity_scores.append(scores)

    # Convert to numpy array for easy indexing
    similarity_scores = np.array(similarity_scores)

    # Find the top N similar HS codes for each 1789 item
    hs_code_matches = []
    for scores in similarity_scores:
        top_indices = np.argsort(-scores)[:top_n]
        top_scores = scores[top_indices]
        top_hs_codes = df_2023.iloc[top_indices]['hts8'].values
        hs_code_matches.append(list(zip(top_hs_codes, top_scores)))

    return hs_code_matches
def match_and_export_hs_codes_spacy(df_1789, df_2023, output_file_path):
    all_matches = find_similar_hs_codes_spacy(df_1789, df_2023)
    
    export_data = []
    for i, matches in enumerate(all_matches):
        for hs_code, score in matches:
            # Find the corresponding 2023 row for the HS code
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
output_csv_path = 'matched_hs_codes_1789_to_2023_spacy.csv'
exported_df_spacy = match_and_export_hs_codes_spacy(df_1789, df_2023, output_csv_path)
print(exported_df_spacy.head())