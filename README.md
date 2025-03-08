# College Student Mental Health Prediction

This repository implements a hierarchical, interpretable machine learning framework to predict mental health status (PHQ-4 categories) using behavioral data from the CES dataset (https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset). The code is organized into multiple modules that mirror the methodology described in the associated paper.


## Overview

The project follows a two-stage hierarchical model:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/1ad34876-5219-47cc-b57b-aadac5ead036" />


1. **Stage 1: Feature Mapping to Interaction Labels**  
   Raw behavioral features are transformed into five interaction labels:  
   - **Leisure**
   - **Me Time**
   - **Phone Time**
   - **Sleep Time**
   - **Social Time**  
   This is achieved by data cleaning, feature engineering, clustering (using KMeans), and personalized feature importance analysis (using Random Forests).

2. **Stage 2: Prediction Using Interaction Labels**  
   The computed interaction label scores are then used as inputs to build personalized neural network models to predict the PHQ-4 mental health categories.


## Setup Instructions

1. **Clone the Repository:**
  ```bash
   git clone https://github.com/roycmeghna/CHASE_2025_Meghna.git
   cd CHASE_2025_Meghna
  ```

2. Create the conda environment:
  ```
  conda env create -f environment.yml
  conda activate my_project_env
  ```
 


2. Run the main script:
  ```
  python main.py
  ```


## Repository Structure

```plaintext
project/
├── main.py                     # Main entry point: loads data, processes domains, and builds models
├── config.py                   # Configuration file (e.g., file paths, constants)
├── data_loader.py              # Module to load and merge raw datasets and add PHQ-4 categorization
├── feature_extractor.py        # Module to extract feature names from the labeled CSV file
├── feature_engineering.py      # Module for cleaning data and adding calculated columns
├── domain_processor.py         # Module to process each interaction label (Leisure, Me Time, Phone Time, Sleep Time, Social Time)
├── model_builder.py            # Module for building and training personalized neural network models per UID
├── utils.py                    # Utility functions (e.g., YAML configuration loader)
├── environment.yml             # Conda environment file listing required packages and Python version
└── README.md                   # This documentation file

```



## Module description 

main.py
  - Main entry point of the project.
  - Loads and preprocesses data.
  - Calls domain processing functions (Leisure, Me Time, Phone Time, Sleep Time, Social Time).
  - Builds and evaluates personalized neural network models.

config.py
  - Stores file paths, constants, and global configurations.

data_loader.py
  - Loads raw datasets, merges them, and applies PHQ-4 categorization.

feature_extractor.py
  - Extracts and organizes feature names from the labeled CSV file.

feature_engineering.py
  - Cleans the data, handles missing values, and adds calculated columns.

domain_processor.py
  - Processes each interaction label:
    - Scales features
    - Applies KMeans clustering to assign behavioral clusters
    - Computes domain-specific scores and personalized feature importances using Random Forests

model_builder.py
  - Builds and trains personalized neural network models per UID.
  - Uses computed interaction label scores as input features.

utils.py
  - Contains general helper functions, such as YAML configuration loaders.

environment.yml
  - Lists required dependencies and the Python version for setting up the environment.



  ## Citing

