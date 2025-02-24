# feature_importance.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

def compute_cluster_score(df, features, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_features)
    df['cluster_score'] = clusters
    # Example: use sum of scaled features plus cluster score as a domain score
    domain_prefix = features[0].split("_")[0]  # e.g., "act" or similar
    df[f'{domain_prefix}_score'] = scaled_features.sum(axis=1) + df['cluster_score']
    return df

def calculate_domain_feature_importance(df, features_rf, uid_column='uid'):
    final_feature_importances = []
    for uid, group in df.groupby(uid_column):
        group[features_rf] = StandardScaler().fit_transform(group[features_rf])
        group[f'{features_rf[0].split("_")[0]}_score'] = group.apply(
            lambda row: sum(1 for col in features_rf if row[col] > 0), axis=1)
        X = group[features_rf]
        y = group[f'{features_rf[0].split("_")[0]}_score']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importance_row = {feature: model.feature_importances_[idx] 
                          for idx, feature in enumerate(features_rf)}
        importance_row['UID'] = uid
        final_feature_importances.append(importance_row)
    final_feature_importances_df = pd.DataFrame(final_feature_importances)
    return final_feature_importances_df

if __name__ == '__main__':
    # Test functions as needed
    pass
