# model_builder.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def build_and_train_model(df, features, target, uid, feature_importances_df):
    uid_data = df[df['uid'] == uid].copy()
    if uid_data.empty:
        print(f"No data for UID: {uid}, skipping...")
        return None, None
    X = uid_data[features].copy()
    y = uid_data[target]
    
    # Retrieve feature weights for this UID if available
    uid_feature_importances = feature_importances_df[feature_importances_df['UID'] == uid]
    if uid_feature_importances.empty:
        print(f"No feature importances for UID: {uid}, skipping...")
        return None, None
    local_feature_weights = uid_feature_importances[features].iloc[0].to_dict()
    
    # Apply feature weights to X
    for feature, weight in local_feature_weights.items():
        X[feature] = X[feature] * weight
    
    X_scaled = StandardScaler().fit_transform(X)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data for UID: {uid} ({e}), skipping...")
        return None, None
    
    y_train_onehot = to_categorical(y_train, num_classes=4)
    y_test_onehot = to_categorical(y_test, num_classes=4)
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    try:
        model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                  epochs=50, batch_size=32, verbose=1)
    except Exception as e:
        print(f"Error training model for UID: {uid} ({e}), skipping...")
        return None, None
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train_onehot, verbose=0)
    results = {
        'UID': uid,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Train Loss': train_loss,
        'Train Accuracy': train_accuracy
    }
    return model, results

def build_models_for_all_uids(df, features, target, feature_importances_df):
    uids = df['uid'].unique()
    uid_models = {}
    results = []
    for uid in uids:
        print(f"Building model for UID: {uid}")
        model, res = build_and_train_model(df, features, target, uid, feature_importances_df)
        if model is not None:
            uid_models[uid] = model
            results.append(res)
    results_df = pd.DataFrame(results)
    return uid_models, results_df

if __name__ == '__main__':
    # For testing purposes
    pass
