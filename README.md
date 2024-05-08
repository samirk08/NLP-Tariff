# NLP-Tariff: Advanced NLP for HS Code Prediction

## Overview
The NLP-Tariff repository is dedicated to the research and development of tools for mapping historical and contemporary U.S. tariff data to Harmonized System (HS) codes using advanced Natural Language Processing (NLP) techniques. This repository not only hosts algorithms for description matching and ad valorem calculations but also explores innovative methods to integrate image processing for tariff classification.

## Repository Structure
This repository is structured to facilitate easy access and understanding of the various components developed during the research:

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


