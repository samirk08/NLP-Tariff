import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
file_2023 = 'tariff database_202305.xlsx'
file_1789 = '1789.xlsx'

df_2023 = pd.read_excel(file_2023)
df_1789 = pd.read_excel(file_1789)
def find_similar_hs_codes(df_1789, df_2023, top_n=1):
    df_1789['combined_description'] = df_1789['item'].fillna('') + ' ' + df_1789['description_1'].fillna('')
    vectorizer = TfidfVectorizer()
    tfidf_matrix_1789 = vectorizer.fit_transform(df_1789['combined_description'])
    tfidf_matrix_2023 = vectorizer.transform(df_2023['brief_description'])
    cosine_similarities = cosine_similarity(tfidf_matrix_1789, tfidf_matrix_2023)
    top_similarities_indices = np.argsort(-cosine_similarities, axis=1)[:, :top_n]

    hs_code_matches = []
    for i in range(top_similarities_indices.shape[0]):
        matched_hs_codes = []
        for j in range(top_n):
            index = top_similarities_indices[i, j]
            similarity_score = cosine_similarities[i, index]
            matched_hs_codes.append((df_2023.iloc[index]['hts8'], df_2023.iloc[index]['brief_description'], similarity_score))
        hs_code_matches.append(matched_hs_codes)

    return hs_code_matches
def match_and_export_hs_codes(df_1789, df_2023, output_file_path):
    all_matches = find_similar_hs_codes(df_1789, df_2023)

    export_data = []
    for i, matches in enumerate(all_matches):
        for hs_code, description, score in matches:
            export_data.append({
                '1789 Item': df_1789.iloc[i]['item'],
                '1789 Description': df_1789.iloc[i]['description_1'],
                'Matched HS Code': hs_code,
                '2023 Description': description,
                'Similarity Score': score
            })

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file_path, index=False)

    return export_df
output_csv_path = 'matched_hs_codes_1789_to_2023.csv'
exported_df = match_and_export_hs_codes(df_1789, df_2023, output_csv_path)
print(exported_df.head())