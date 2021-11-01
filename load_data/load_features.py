# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:20:50 2020

@author: hersh.ravkin
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# csv_path = "H:/Manager/datasets/v6-no-censored.csv"


def extract_features(df):
    print("\nExtracting Features")
    # replace "censored" with None
    df = df.replace("censored", None)
    # Drop columns where all entries are NaN
    features = df.dropna(axis=1, how='all')
    print("\nNumber of columns dropped because all entries were NaN: ",
          len(df.columns)-len(features.columns))
    print("Columns dropped:")
    print(list(set(df.columns) - set(features.columns)))
    # Drop target columns
    target_cols = features.columns[features.columns.str.startswith('target')]
    features = features.drop(target_cols, axis=1)
    # don't use decease date (leakage) and complications risk description (need to check if okay to use or not)
    print("removing known leakage columns from features")
    features = features.drop(['Reference Event-Complications risk description',
                              'Reference Event-Decease date-Days from Reference',
                              'first nurse questionnaire-Event date-Days from Reference'
                              ],
                             axis=1, errors='ignore')
    constant_cols = features.columns[features.nunique() <= 1]
    print("Dropping constant columns:")
    print(constant_cols)
    features = features.drop(constant_cols, axis=1)
    # Select age columns to be OHE
    agecols_to_ohe = [col for col in features.columns if 'Age at' in col]
    agecols_to_ohe.remove('Reference Event-Age at event')
    # agecols_to_ohe.remove('Alcohol Dependance Syndrome-Age at diagnosis')
    ageohe_cols = [(col.split("-Age")[0]+" AGE_OHE") for col in agecols_to_ohe]
    #Select default date columns to be OHE
    datecols_to_ohe = [col for col in features.columns if 'Diagnosis date' in col]
    # datecols_to_ohe.remove('Alcohol Dependance Syndrome-Diagnosis date-Days from Reference')
    dateohe_cols = [(col.split("-Diagnosis")[0]+" DATE_OHE") for col in datecols_to_ohe]
    
    #Combine age and date calls for OHE
    ageohe_cols.extend(dateohe_cols)
    agecols_to_ohe.extend(datecols_to_ohe)
    
    #Create ohe_cols based on entry for age columns being not null
    features[ageohe_cols] = features[agecols_to_ohe].applymap(lambda x: 1 if np.isfinite(x) else 0)
    
    #diagnosis_cols = [col for col in features.columns if 'Diagnosis Code Key' in col]
    
    #Select categorical features and fill NaNs
    categorical_feats = features.select_dtypes(include='object').columns
    features = not_diagnosed_fill(features, categorical_feats)
    #sys.exit()
    #Encode categorical features
    features = encode_categoricals(features, categorical_feats)
    features = not_diagnosed_fill(features, ageohe_cols)
    return features


def not_diagnosed_fill(df, ohe_cols):
    df[ohe_cols] = df[ohe_cols].fillna(value='NOT DIAGNOSED')
    return df


def encode_categoricals(df, categorical_feats):
    print("encoding categorical features:")
    print(categorical_feats)
    strategy = LabelEncoder()
    df[categorical_feats] = df[categorical_feats].apply(strategy.fit_transform)
    return df


def load_features(csv_path):
    df_features = pd.read_csv(csv_path)
    # df_features = df_features[df_features['Reference Event-Age at event'] > 60]
    df_features = extract_features(df_features)
    return df_features

# load_features(csv_path)
