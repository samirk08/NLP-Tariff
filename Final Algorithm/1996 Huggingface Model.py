import pandas as pd
def process_files(gpt_sample_path, hf_sample_path, output_path):
    # Load the files
    gpt_df = pd.read_csv(gpt_sample_path)
    hf_df = pd.read_csv(hf_sample_path)

    # Filter out "Missing description" and "Other" in "1996 Item" column
    gpt_df_filtered = gpt_df[~gpt_df['1996 Item'].str.contains("Missing description|Other", case=False, na=False)]
    hf_df_filtered = hf_df[~hf_df['1996 Item'].str.contains("Missing description|Other", case=False, na=False)]

    # Sort the filtered dataframes
    gpt_df_filtered_sorted = gpt_df_filtered.sort_values(by="1996 Item")
    hf_df_filtered_sorted = hf_df_filtered.sort_values(by="1996 Item")

    # Merge the filtered and sorted dataframes on "1996 Item"
    merged_filtered_df = pd.merge(gpt_df_filtered_sorted, hf_df_filtered_sorted, on="1996 Item", suffixes=('_GPT', '_HF'))

    # Determine the higher confidence score for each item
    merged_filtered_df['Higher Confidence Source'] = merged_filtered_df.apply(
        lambda x: 'GPT' if x['Confidence Score_GPT'] > x['Confidence Score_HF'] else 'HF', axis=1
    )

    # Include a column for the final confidence level
    merged_filtered_df['Final Confidence Level'] = merged_filtered_df.apply(
        lambda x: x['Confidence Score_GPT'] if x['Higher Confidence Source'] == 'GPT' else x['Confidence Score_HF'], axis=1
    )

    # Select data based on the higher confidence score
    final_filtered_df = merged_filtered_df[['1996 Item', 'Predicted HS Code_GPT', 'Predicted HS Code_HF', 'Higher Confidence Source', 'Final Confidence Level']].copy()
    final_filtered_df['Final Predicted HS Code'] = merged_filtered_df.apply(
        lambda x: x['Predicted HS Code_GPT'] if x['Higher Confidence Source'] == 'GPT' else x['Predicted HS Code_HF'], axis=1
    )

    # Drop the separate HS Code columns
    final_filtered_df = final_filtered_df.drop(['Predicted HS Code_GPT', 'Predicted HS Code_HF'], axis=1)

    # Save to the specified output path
    final_filtered_df.to_csv(output_path, index=False)

    return output_path
output_path = process_files('/home/samirk08/UROP_SPRING_2024/1996/1996_GPT_Samlpe.csv', '/home/samirk08/UROP_SPRING_2024/1996/HF_1996_Sample.csv', '1996_SAMPLE_HYBRID.csv')
print(f"Output saved to: {output_path}")