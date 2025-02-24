# feature_extractor.py

import pandas as pd
from config import LABELED_FILE

def extract_feature_names():
    df1 = pd.read_csv(LABELED_FILE)
    features_leisure = []
    features_me = []
    features_phone = []
    features_sleep = []
    features_social = []
    for _, row in df1.iterrows():
        labels = row['Labels']
        variable = row['Variable']
        if isinstance(labels, str):
            try:
                labels = eval(labels)
            except Exception as e:
                print(f"Error parsing labels for {variable}: {e}")
                labels = []
        if isinstance(labels, list):
            if 0 in labels:
                features_leisure.append(variable)
            if 1 in labels:
                features_me.append(variable)
            if 2 in labels:
                features_phone.append(variable)
            if 3 in labels:
                features_sleep.append(variable)
            if 4 in labels:
                features_social.append(variable)
    return {
        "leisure": features_leisure,
        "me": features_me,
        "phone": features_phone,
        "sleep": features_sleep,
        "social": features_social
    }

if __name__ == '__main__':
    features = extract_feature_names()
    print(features)
