# **I-HOPE** — Interpretable Hierarchical mOdel for Personalized mEntal Health Prediction

This repository contains code for **I-HOPE** — **I**nterpretable **H**ierarchical m**O**del for **P**ersonalized m**E**ntal Health Prediction. I-HOPE is a mental health prediction system that employs a two-stage hierarchical model to map raw behavioral features to mental health status (PHQ-4 categories). It does so by leveraging five defined behavioral categories, referred to as *interaction labels*. This work utilizes the CES dataset(https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset).

## Overview

The project follows a two-stage hierarchical model as shown below:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/aae9cc55-3b38-4076-89aa-022b8de2da52" />

1. **Stage 1: Feature Mapping to Interaction Labels**  
   35 chosen raw behavioral features are transformed into five interaction labels:  
   - **Leisure**
   - **Me Time**
   - **Phone Time**
   - **Sleep**
   - **Social Time**  
   This is achieved by data cleaning, feature engineering, clustering (using KMeans), and personalized feature importance analysis (using Random Forests).

### Selected Features and Relevant Labels

| **#** | **Feature Name** | **Relevant Labels[0: Leisure, 1: MeTime, 2: Phone, 3: Sleep , 4: SocialInt]** |
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

## Files
   1. Github_code.ipynb -- Notebook with the code
   2. Csvs used --
      (a) Initial dataset: Got after running Section 2(a) of https://github.com/bill-wei-xuan/Unlocking-MentalHealth/blob/main/notebooks/DigitalWellbeing.ipynb
      (b) Manual labels : CHASE_Labeled.csv
      All the original data can be downloaded from https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset


