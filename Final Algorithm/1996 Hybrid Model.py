import pandas as pd

def process_files(gpt_sample_path, hf_sample_path, output_path):
    """
    Processes and merges two datasets based on confidence scores from GPT and HF models.

    Args:
        gpt_sample_path (str): File path for the GPT model output data.
        hf_sample_path (str): File path for the HF model output data.
        output_path (str): File path where the merged data will be saved.

    Returns:
        str: The output file path where the processed data is saved.
    """
    # Load GPT and HF data from CSV files
    gpt_df = pd.read_csv(gpt_sample_path)
    hf_df = pd.read_csv(hf_sample_path)

    # Filter entries where "1996 Item" contains "Missing description" or "Other"
    gpt_df_filtered = gpt_df[~gpt_df['1996 Item'].str.contains("Missing description|Other", case=False, na=False)]
    hf_df_filtered = hf_df[~hf_df['1996 Item'].str.contains("Missing description|Other", case=False, na=False)]

    # Sort the filtered dataframes by "1996 Item" for consistent merging
    gpt_df_filtered_sorted = gpt_df_filtered.sort_values(by="1996 Item")
    hf_df_filtered_sorted = hf_df_filtered.sort_values(by="1996 Item")

    # Merge the filtered and sorted dataframes on "1996 Item"
    merged_filtered_df = pd.merge(gpt_df_filtered_sorted, hf_df_filtered_sorted, on="1996 Item", suffixes=('_GPT', '_HF'))

    # Determine the source with the higher confidence score for each item
    merged_filtered_df['Higher Confidence Source'] = merged_filtered_df.apply(
        lambda x: 'GPT' if x['Confidence Score_GPT'] > x['Confidence Score_HF'] else 'HF', axis=1
    )

    # Create a column for the final confidence level based on the higher score
    merged_filtered_df['Final Confidence Level'] = merged_filtered_df.apply(
        lambda x: x['Confidence Score_GPT'] if x['Higher Confidence Source'] == 'GPT' else x['Confidence Score_HF'], axis=1
    )

    # Extract the final predicted HS Code based on the higher confidence score
    merged_filtered_df['Final Predicted HS Code'] = merged_filtered_df.apply(
        lambda x: x['Predicted HS Code_GPT'] if x['Higher Confidence Source'] == 'GPT' else x['Predicted HS Code_HF'], axis=1
    )

    # Select relevant columns and drop the separate HS Code columns
    final_filtered_df = merged_filtered_df[['1996 Item', 'Higher Confidence Source', 'Final Predicted HS Code', 'Final Confidence Level']].copy()

    # Save the final merged dataset to the specified output path
    final_filtered_df.to_csv(output_path, index=False)

    return output_path

# Define the file paths
gpt_sample_path = '/path/to/1996_GPT_Sample.csv'
hf_sample_path = '/path/to/HF_1996_Sample.csv'
output_path = 'path/to/1996_SAMPLE_HYBRID.csv'

# Process the files and output the location where the final data is saved
output_path = process_files(gpt_sample_path, hf_sample_path, output_path)
print(f"Output saved to: {output_path}")
