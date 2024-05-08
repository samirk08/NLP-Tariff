# NLP-Tariff: Advanced NLP for HS Code Prediction

## Overview
The NLP-Tariff repository is dedicated to the research and development of tools for mapping historical and contemporary U.S. tariff data to Harmonized System (HS) codes using advanced Natural Language Processing (NLP) techniques. This repository not only hosts algorithms for description matching and HS code matching but also explores innovative methods to integrate image processing for tariff classification.

## Repository Structure
This repository is structured to facilitate easy access and understanding of the various components developed during the research:

### Tariff Predictions
All tariff predictions within this repository are based on the **2023 Tariff Database**. This ensures that all HS code mappings and related calculations are up-to-date with the most current global trade compliance regulations and standards.

### Algorithms
- **Huggingface Algorithm**: Utilizes Huggingface's transformers for generating embeddings that enhance the accuracy of HS code predictions from product descriptions.
- **GPT Algorithm**: Implements OpenAI's GPT models to refine and generate descriptive text that aids in aligning product descriptions with their respective HS codes more accurately.
- **Hybrid Model**: Combines the strengths of both Huggingface and GPT models to maximize prediction accuracy. This model leverages the detailed embeddings from Huggingface with the contextual understanding of GPT, providing a robust solution for HS code prediction.

### Final Algorithm
The final algorithm represents a culmination of extensive research and development, incorporating:
- **Embeddings.ipynb**: A Jupyter notebook that details the process of generating and utilizing embeddings for HS code prediction.
- **1996 Models**: Includes separate models (Huggingface, GPT, Hybrid) tailored to predict HS codes for the year 1996, showcasing the evolution and iterative improvement of our prediction algorithms.

### Picture to Tariff
This innovative component extends the repository's capabilities to include image processing:
- **Image to Tariff.ipynb**: A notebook that demonstrates the conversion of images to tariff codes by first extracting text descriptions from images and then using NLP models to predict HS codes.
- **Image to Tariff.py**: Provides a script version of the notebook for operational use, allowing images to be directly converted into HS codes through automated scripts.

## Getting Started
To get started with the NLP-Tariff repository, clone the repo and install the required dependencies:
```bash
git clone https://github.com/your-github-username/nlp-tariff.git
cd nlp-tariff
pip install -r requirements.txt
