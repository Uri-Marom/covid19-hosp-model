import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def extract_features(df):
    target_cols = df.columns.str.startswith('target')
    features = df.dropna(axis=1, how='all')
    features = df.drop(target_cols, axis=1)

    diagnosis_cols = features.columns[features.columns.str.endswith('Diagnosis Code Key')]
    
    features = not_diagnosed_fill(features, diagnosis_cols)
    return features


def not_diagnosed_fill(df, diagnosis_cols):
    df[diagnosis_cols] = df[diagnosis_cols].fillna(value='NOT DIAGNOSED')
    return df


def load_features(features_csv):
    df = pd.read_csv(features_csv)
    return extract_features(df)
