# GPT Algorithm Directory

This directory contains files related to the GPT algorithm used in the NLP-Tariff project for advanced NLP processing. Here is a detailed overview of each file, its functionalities, and how parallel processing and threading have been utilized.

## Files Overview

### GPT Algorithm.py

- **Description**: This Python script contains the core GPT algorithm implementation. The code initializes the GPT model, configures its parameters, and sets up data pipelines for processing.
- **Technical Details**:
  - **Parallel Processing**: The script utilizes multi-threading and parallel processing techniques to enhance performance. It employs Python's `concurrent.futures` module to manage a pool of threads, allowing simultaneous processing of multiple data batches.
  - **Data Handling**: Utilizes efficient data handling mechanisms to manage memory usage when loading large datasets, possibly leveraging generators or async I/O operations.
  - **Usage**: Essential for developers and researchers who need to modify the AI model or understand its detailed workings.

### GPT Example - 1990.py

- **Description**: This Python script demonstrates the application of the GPT algorithm to tariff data from the year 1990. It includes data loading, model execution, and output handling.
- **Usage**: Run this script to see how the GPT model processes specific historical data and generates outputs based on the model predictions.

### GPT4_1990_F1Scores.csv

- **Description**: A CSV file containing F1 scores and other relevant performance metrics for the GPT model outputs corresponding to the year 1990.
- **Usage**: Review this file to assess model performance and for use in analytical comparisons or further statistical analysis.

### GPT4_1990_Results.csv

- **Description**: This CSV file lists the output results of the GPT model for the year 1990, detailing predictions, confidence scores, and other metrics that help in evaluating the accuracy and reliability of the model.
- **Usage**: Can be directly used for detailed analysis or integrated with other datasets for comprehensive data analysis projects.

## Additional Information

For detailed instructions on setting up the environment to run these scripts, please refer to the `requirements.txt` or `environment.yml` files in the main directory. Any issues or suggestions can be addressed by opening an issue in this repository or contacting the project maintainers directly.
