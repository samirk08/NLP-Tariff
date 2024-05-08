import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score


# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

file_2023 = 'tariff database_202305.xlsx'
file_1990 = '1990 WIP AZ.xlsx' 

df_2023 = pd.read_excel(file_2023)
df_1990 = pd.read_excel(file_1990)  # Load 1990 data


def find_similar_hs_codes_transformers(df_1990, df_2023, top_n=1):
    # Combining item and description fields for 1990 data
    df_1990['combined_description'] = df_1990['ProductDescription'].fillna('')

    # Compute embeddings for each description
    embeddings_1990 = model.encode(df_1990['combined_description'].tolist(), convert_to_tensor=True)
    embeddings_2023 = model.encode(df_2023['brief_description'].tolist(), convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(embeddings_1990, embeddings_2023)

    # Find the top N similar HS codes for each 1990 item
    hs_code_matches = []
    for i in range(len(df_1990)):
        top_results = np.argsort(-cosine_scores[i].cpu().numpy())[:top_n]
        matched_hs_codes = [(df_2023.iloc[j]['hts8'], cosine_scores[i][j].item()) for j in top_results]
        hs_code_matches.append(matched_hs_codes)

    return hs_code_matches

def calculate_f1_scores(y_true, y_pred, digit_level=10):
    # Truncate HS codes to the specified digit level
    y_true_truncated = [str(code)[:digit_level] for code in y_true]
    y_pred_truncated = [str(code)[:digit_level] for code in y_pred]

    # Calculate F1 score
    return f1_score(y_true_truncated, y_pred_truncated, average='weighted')

def match_and_export_hs_codes_transformers(df_1990, df_2023, output_file_path):
    all_matches = find_similar_hs_codes_transformers(df_1990, df_2023)

    export_data = []
    for i, matches in enumerate(all_matches):
        for hs_code, score in matches:
            df_2023_row = df_2023[df_2023['hts8'] == hs_code].iloc[0]
            original_hs_code = df_1990.iloc[i]['ProductCode']  # Replace with actual column name
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



output_csv_path = 'matched_hs_codes_1990_to_2023_transformers.csv'
exported_df_transformers = match_and_export_hs_codes_transformers(df_1990, df_2023, output_csv_path)
print(exported_df_transformers.head())
