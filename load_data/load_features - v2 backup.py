# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:20:50 2020

@author: hersh.ravkin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#csv_path = "C:/Users/hersh.ravkin/Downloads/V5.synthetic.csv"



def extract_features(df):
    # replace "censored" with None
    df = df.replace("censored", None)
    # Drop columns where all entries are NaN
    features = df.dropna(axis=1, how='all')
    print("\nNumber of columns dropped because all entries were NaN: ",
          len(df.columns)-len(features.columns))
    # Drop target columns
    target_cols = features.columns[features.columns.str.startswith('target')]
    features = features.drop(target_cols, axis=1)
    # Temporary specific drop (shouldn't have been included in the file)
    features = features.drop(['Cancer Diagnosis 2 - last diagnosis-Age at diagnosis',
                              'Cancer Diagnosis 2 - first diagnosis-Age at diagnosis',
                              'Reference Event-Complications risk description',
                              'Reference Event-Decease date', 'Reference Event-Event date',
                              'Cancer Diagnosis MAC-Status in registry',
                              'Aspiration Pneumonia-Diagnosis Code Key',
                              ],
                       axis=1)
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
    # TEMPORARY - dropping cat features
    features = features.drop(categorical_feats, axis=1)
    # skipping for now, later error: could not convert string to float: 'NOT DIAGNOSED'
    # features = not_diagnosed_fill(features, categorical_feats)
    #Encode categorical features
    # skipping for now, TypeError: Encoders require their input to be uniformly strings or numbers. Got ['float', 'str']
    # features = encode_categoricals(features, categorical_feats)
    # skipping for now, later error: could not convert string to float: 'NOT DIAGNOSED'
    # features = not_diagnosed_fill(features, ageohe_cols)
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
    df = pd.read_csv(csv_path)
    return extract_features(df)

