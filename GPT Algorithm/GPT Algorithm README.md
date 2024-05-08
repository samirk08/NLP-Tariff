# GPT Algorithm Directory

This directory contains files related to the GPT algorithm used in the NLP-Tariff project for advanced NLP processing. Here is a detailed overview of each file, its functionalities, and how parallel processing and threading have been utilized.

## Files Overview

### GPT Algorithm.py

- **Description**: This Python script contains the core GPT algorithm implementation. The code initializes the GPT model, configures its parameters, and sets up data pipelines for processing.
- **Technical Details**:
  - **SentenceTransformer and FuzzyWuzzy**: Utilizes SentenceTransformer for semantic text embeddings and FuzzyWuzzy for calculating textual similarities, enabling effective matching of HS codes.
  - **Parallel Processing**: Employs `ThreadPoolExecutor` from `concurrent.futures` to enhance computational efficiency by processing multiple data entries concurrently.
  - **Error Handling and Retries**: Implements robust error handling with `tenacity` for automatic retrying of API requests, using an exponential back-off strategy.
  - **NLP and API Integration**: Enhances descriptions and performs complex queries using OpenAI's GPT, directly interfacing through the OpenAI Python client to fetch and calculate the most accurate HS     codes.

### GPT Example - 1990.py

- **Description**: This Python script demonstrates the application of the GPT algorithm to tariff data from the year 1990. It includes data loading, model execution, and output handling.

### GPT4_1990_F1Scores.csv

- **Description**: A CSV file containing F1 scores and other relevant performance metrics for the GPT model outputs corresponding to the year 1990.
- **Usage**: Review this file to assess model performance and for use in analytical comparisons or further statistical analysis.

### GPT4_1990_Results.csv

- **Description**: This CSV file lists the output results of the GPT model for the year 1990, detailing predictions, confidence scores, and other metrics that help in evaluating the accuracy and reliability of the model.
- **Usage**: Can be directly used for detailed analysis or integrated with other datasets for comprehensive data analysis projects.


