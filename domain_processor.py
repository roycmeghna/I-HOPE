# domain_processor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

def process_domain(data, features_rf, domain_label, n_clusters=3, random_state=42):
    """
    Processes a given domain by:
    - Scaling the selected raw features (features_rf)
    - Applying KMeans clustering to generate a cluster score
    - Creating an intermediate domain score as the sum of scaled features plus cluster score
    - Computing personalized feature importance via RandomForestRegressor on a per-UID basis

    Returns:
        data_domain: DataFrame with an added column for the domain score (e.g., leisure_score)
        importance_df: DataFrame of personalized feature importances for the domain
    """
    if not features_rf:
        print(f"No features provided for domain {domain_label}.")
        return data, None
    
    data_domain = data.copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_domain[features_rf])
    
    # Clustering to capture intra-domain behavioral patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_features)
    data_domain['cluster_score'] = clusters
    
    # Create a preliminary domain score (for example, sum of scaled features + cluster score)
    data_domain[f'{domain_label}_score'] = scaled_features.sum(axis=1) + data_domain['cluster_score']
    
    # Compute personalized feature importance per UID
    final_feature_importances = []
    for uid, group in data_domain.groupby('uid'):
        group_features = group[features_rf]
        # Standardize features for the group
        group_scaled = scaler.fit_transform(group_features)
        # Define a temporary score as the count of features with a positive deviation (above mean)
        group['temp_score'] = group.apply(lambda row: sum(1 for col in features_rf if row[col] > 0), axis=1)
        X = group[features_rf]
        y = group['temp_score']
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X, y)
        importance_row = {feature: model.feature_importances_[idx] for idx, feature in enumerate(features_rf)}
        importance_row['UID'] = uid
        final_feature_importances.append(importance_row)
    importance_df = pd.DataFrame(final_feature_importances)
    return data_domain, importance_df

def process_leisure(data, leisure_features_rf, n_clusters=3):
    return process_domain(data, leisure_features_rf, 'leisure', n_clusters=n_clusters)

def process_me_time(data, me_features_rf, n_clusters=6):
    return process_domain(data, me_features_rf, 'metime', n_clusters=n_clusters)

def process_phone_time(data, phone_features_rf, n_clusters=3):
    return process_domain(data, phone_features_rf, 'phone', n_clusters=n_clusters)

def process_sleep_time(data, sleep_features_rf, n_clusters=3):
    return process_domain(data, sleep_features_rf, 'sleep', n_clusters=n_clusters)

def process_social_time(data, social_features_rf, n_clusters=3):
    return process_domain(data, social_features_rf, 'social', n_clusters=n_clusters)

if __name__ == '__main__':
    # For testing purposes
    pass
