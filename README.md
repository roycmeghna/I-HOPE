# **I-HOPE** — **_I_*nterpretable Hierarchical m**_O_*del for **_P_*ersonalized m**_E_*ntal Health Prediction



This repository contains code for **I-HOPE** — **I**nterpretable **H**ierarchical m**O**del for **P**ersonalized m**E**ntal Health Prediction. I-HOPE is a mental health prediction system that employs a two-stage hierarchical model to map raw behavioral features to mental health status (PHQ-4 categories). It does so by leveraging five defined behavioral categories, referred to as *interaction labels*. This work utilizes the CES dataset(https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset).

## Overview

The project follows a two-stage hierarchical model as shown below:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/aae9cc55-3b38-4076-89aa-022b8de2da52" />

1. **Stage 1: Feature Mapping to Interaction Labels**  
   35 chosen raw behavioral features are transformed into five interaction labels:  
   - **Leisure**
   - **Me Time**
   - **Phone Time**
   - **Sleep Time**
   - **Social Time**  
   This is achieved by data cleaning, feature engineering, clustering (using KMeans), and personalized feature importance analysis (using Random Forests).

### Selected Features and Relevant Labels

| **#** | **Feature Name** | **Labels** |
|----|-----------------|-----------|
| 1  | act_on_bike_ep_0 | [0,1] |
| 2  | act_on_foot_ep_0 | [0,1,4] |
| 3  | act_running_ep_0 | [0,1] |
| 4  | act_still_ep_0 | [1,3] |
| 5  | act_walking_ep_0 | [0,1,4] |
| 6  | audio_convo_duration_ep_0 | [0,2,4] |
| 7  | (call_in_num + call_out_num) / (call_in_duration + call_out_duration) | [0,2] |
| 8  | loc_food_audio_voice | [4] |
| 9  | loc_home_audio_voice | [1,2,3] |
| 10 | loc_social_audio_voice | [0,2,4] |
| 11 | loc_other_dorm_audio_voice | [0,4] |
| 12 | loc_self_dorm_audio_voice | [1,2] |
| 13 | loc_study_audio_voice | [1,4] |
| 14 | loc_food_convo_duration | [4] |
| 15 | loc_home_convo_duration | [1,2,3] |
| 16 | loc_other_dorm_convo_duration | [0,4] |
| 17 | loc_social_convo_duration | [0,4] |
| 18 | loc_study_convo_duration | [1,4] |
| 19 | loc_self_dorm_convo_duration | [1,2] |
| 20 | loc_home_dur | [1,3] |
| 21 | loc_leisure_dur | [0,4] |
| 22 | loc_other_dorm_dur | [4] |
| 23 | loc_self_dorm_dur | [1,3] |
| 24 | loc_social_dur | [4] |
| 25 | loc_study_dur | [1,4] |
| 26 | loc_workout_dur | [0,1] |
| 27 | loc_home_unlock_num / loc_home_unlock_duration | [0,1,2] |
| 28 | loc_other_dorm_unlock_num / loc_other_dorm_unlock_duration | [0,4,2] |
| 29 | loc_self_dorm_unlock_num / loc_self_dorm_unlock_duration | [1,2] |
| 30 | loc_social_unlock_num / loc_social_unlock_duration | [2,4] |
| 31 | loc_study_unlock_num / loc_study_unlock_duration | [1,2,3] |
| 32 | sleep_duration | [3] |
| 33 | sleep_end - sleep_start | [3] |
| 34 | unlock_num_ep_0 / unlock_duration_ep_0 | [2] |
| 35 | sleep_heathkit_dur | [3] |


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

