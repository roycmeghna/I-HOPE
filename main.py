# main.py

from data_loader import load_and_merge_data, add_phq4_category
from feature_extractor import extract_feature_names
from feature_engineering import fill_missing_values, add_calculated_columns
from domain_processor import (
    process_leisure, process_me_time, process_phone_time,
    process_sleep_time, process_social_time
)
from model_builder import build_models_for_all_uids
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def main():
    # Load and preprocess the merged dataset
    merged = load_and_merge_data()
    merged = add_phq4_category(merged)
    
    # Select relevant columns (adjust the list as needed)
    selected_columns = [
        'uid', 'is_ios', 'day',
        'act_on_bike_ep_0', 'act_on_foot_ep_0', 'act_running_ep_0', 'act_still_ep_0', 'act_walking_ep_0',
        'audio_convo_duration_ep_0',
        'call_in_num_ep_0', 'call_out_num_ep_0', 'call_in_duration_ep_0', 'call_out_duration_ep_0',
        'loc_food_audio_voice', 'loc_home_audio_voice', 'loc_social_audio_voice',
        'loc_other_dorm_audio_voice', 'loc_self_dorm_audio_voice', 'loc_study_audio_voice',
        'loc_food_convo_duration', 'loc_home_convo_duration', 'loc_other_dorm_convo_duration',
        'loc_social_convo_duration', 'loc_study_convo_duration', 'loc_self_dorm_convo_duration',
        'loc_home_dur', 'loc_leisure_dur', 'loc_other_dorm_dur', 'loc_self_dorm_dur', 'loc_social_dur',
        'loc_study_dur', 'loc_workout_dur',
        'loc_home_unlock_num', 'loc_home_unlock_duration',
        'loc_other_dorm_unlock_num', 'loc_other_dorm_unlock_duration',
        'loc_self_dorm_unlock_num', 'loc_self_dorm_unlock_duration',
        'loc_social_unlock_num', 'loc_social_unlock_duration',
        'loc_study_unlock_num', 'loc_study_unlock_duration',
        'sleep_duration', 'sleep_end', 'sleep_start',
        'unlock_num_ep_0', 'unlock_duration_ep_0',
        'sleep_heathkit_dur', 'phq4_category'
    ]
    data = merged[selected_columns].copy()
    data = fill_missing_values(data)
    data = add_calculated_columns(data)
    
    # Extract feature names from the labeled data (if needed)
    features_dict = extract_feature_names()
    
    # Define raw feature lists for each domain (adjust as needed)
    leisure_features_rf = [
         'act_on_bike_ep_0', 'act_on_foot_ep_0', 'act_running_ep_0', 'act_walking_ep_0',
         'audio_convo_duration_ep_0',
         '(call_in_num_ep_0+call_out_num_ep_0) /(call_in_duration_ep_0+call_out_duration_ep_0)',
         'loc_social_audio_voice', 'loc_other_dorm_audio_voice', 'loc_other_dorm_convo_duration',
         'loc_social_convo_duration', 'loc_leisure_dur', 'loc_workout_dur',
         'loc_home_unlock_num/loc_home_unlock_duration', 'loc_other_dorm_unlock_num/ loc_other_dorm_unlock_duration'
    ]
    me_features_rf = [
         'act_on_bike_ep_0', 'act_on_foot_ep_0', 'act_running_ep_0', 'act_still_ep_0', 'act_walking_ep_0',
         'loc_home_audio_voice', 'loc_self_dorm_audio_voice', 'loc_study_audio_voice',
         'loc_home_convo_duration', 'loc_study_convo_duration', 'loc_self_dorm_convo_duration',
         'loc_self_dorm_dur', 'loc_study_dur', 'loc_workout_dur',
         'loc_home_unlock_num/loc_home_unlock_duration', 'loc_self_dorm_unlock_num/ loc_self_dorm_unlock_duration',
         'loc_study_unlock_num/loc_study_unlock_duration'
    ]
    phone_features_rf = [
         'audio_convo_duration_ep_0',
         '(call_in_num_ep_0+call_out_num_ep_0) /(call_in_duration_ep_0+call_out_duration_ep_0)',
         'loc_home_audio_voice', 'loc_social_audio_voice', 'loc_self_dorm_audio_voice',
         'loc_home_convo_duration', 'loc_self_dorm_convo_duration',
         'loc_home_unlock_num/loc_home_unlock_duration', 'loc_other_dorm_unlock_num/ loc_other_dorm_unlock_duration',
         'loc_self_dorm_unlock_num/ loc_self_dorm_unlock_duration', 'loc_social_unlock_num/ loc_social_unlock_duration',
         'loc_study_unlock_num/loc_study_unlock_duration', 'unlock_num_ep_0/ unlock_duration_ep_0'
    ]
    sleep_features_rf = [
         'act_still_ep_0', 'loc_home_audio_voice', 'loc_home_convo_duration',
         'loc_self_dorm_dur', 'loc_study_unlock_num/loc_study_unlock_duration',
         'sleep_duration', 'sleep_end - sleep_start', 'sleep_heathkit_dur'
    ]
    social_features_rf = [
         'act_on_foot_ep_0', 'act_walking_ep_0', 'audio_convo_duration_ep_0',
         'loc_food_audio_voice', 'loc_social_audio_voice', 'loc_other_dorm_audio_voice',
         'loc_study_audio_voice', 'loc_food_convo_duration', 'loc_other_dorm_convo_duration',
         'loc_social_convo_duration', 'loc_study_convo_duration', 'loc_leisure_dur',
         'loc_other_dorm_dur', 'loc_social_dur', 'loc_study_dur',
         'loc_other_dorm_unlock_num/ loc_other_dorm_unlock_duration', 'loc_social_unlock_num/ loc_social_unlock_duration'
    ]
    
    # Process each domain separately
    data_leisure, leisure_importances_df = process_leisure(data, leisure_features_rf)
    data_me, me_importances_df = process_me_time(data, me_features_rf)
    data_phone, phone_importances_df = process_phone_time(data, phone_features_rf)
    data_sleep, sleep_importances_df = process_sleep_time(data, sleep_features_rf)
    data_social, social_importances_df = process_social_time(data, social_features_rf)
    
    # For demonstration, assume that each domain function adds a new score column (e.g., leisure_score, metime_score, etc.)
    # Here we collect these domain scores into a final feature set.
    domain_score_columns = ['leisure_score', 'metime_score', 'phone_score', 'sleep_score', 'social_score']
    
    # Optionally, one may merge or verify these columns exist in the final data.
    # Build final models for each UID using the domain scores as inputs.
    final_features = domain_score_columns
    target = 'phq4_category'
    # For model building, we could choose one of the importance DataFrames.
    uid_models, results_df = build_models_for_all_uids(data, final_features, target, leisure_importances_df)
    print("Evaluation Results:")
    print(results_df)
    results_df.to_csv('Accuracy.csv', index=False)
    
if __name__ == '__main__':
    main()
