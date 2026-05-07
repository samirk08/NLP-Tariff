import pandas as pd
from sklearn.metrics import f1_score
path_new_gpt4_results = '/Users/samirkadariya/Desktop/School/UROP IAP 2024/GPT/1990 - GPT4/GPT Output.csv'
df_new_gpt4_results = pd.read_csv(path_new_gpt4_results)

# Extracting relevant columns
df_new_gpt4_results['Matched HS Code'] = df_new_gpt4_results['Matched HS Code'].astype(str)
df_new_gpt4_results['1990 Product Code'] = df_new_gpt4_results['1990 Product Code'].astype(str)

def calculate_binary_f1_scores(y_true, y_pred, digit_level):
    y_true_truncated = [code[:digit_level] for code in y_true]
    y_pred_truncated = [code[:digit_level] for code in y_pred]
    matches = [1 if true_code == pred_code else 0 for true_code, pred_code in zip(y_true_truncated, y_pred_truncated)]
    return f1_score(matches, [1]*len(matches), zero_division=1)

# Calculating F1 scores for HS4, HS6, and HS8 as binary metrics
f1_hs4_binary = calculate_binary_f1_scores(df_new_gpt4_results['Matched HS Code'], df_new_gpt4_results['1990 Product Code'], 4)
f1_hs6_binary = calculate_binary_f1_scores(df_new_gpt4_results['Matched HS Code'], df_new_gpt4_results['1990 Product Code'], 6)
f1_hs8_binary = calculate_binary_f1_scores(df_new_gpt4_results['Matched HS Code'], df_new_gpt4_results['1990 Product Code'], 8)

f1_hs4_binary, f1_hs6_binary, f1_hs8_binary
def add_individual_f1_scores(df, y_true, y_pred):
    # Adding individual F1 scores for each row in the dataframe
    df['F1 Score HS4'] = [f1_score([1], [1 if true_code[:4] == pred_code[:4] else 0], zero_division=1) for true_code, pred_code in zip(y_true, y_pred)]
    df['F1 Score HS6'] = [f1_score([1], [1 if true_code[:6] == pred_code[:6] else 0], zero_division=1) for true_code, pred_code in zip(y_true, y_pred)]
    df['F1 Score HS8'] = [f1_score([1], [1 if true_code[:8] == pred_code[:8] else 0], zero_division=1) for true_code, pred_code in zip(y_true, y_pred)]
    return df

# Adding individual F1 scores to the dataframe
df_new_gpt4_results_with_f1 = add_individual_f1_scores(df_new_gpt4_results, df_new_gpt4_results['Matched HS Code'], df_new_gpt4_results['1990 Product Code'])

# Exporting the dataframe with F1 scores to a new CSV file
new_export_path = 'Jan25.csv'
df_new_gpt4_results_with_f1.to_csv(new_export_path, index=False)

new_export_path