# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:20:50 2020

@author: hersh.ravkin
"""

import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import load_features
import load_target_variable
#import missingno as msno
import featuretools as ft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

csv_path = "H:/Manager/datasets/v7.1.csv"

"""
Dataset Generation (calling the above code)
"""
target_col = load_target_variable.load_target_variable(csv_path)
features_df = load_features.load_features(csv_path)
#QA: check out whats going on with 'Asthma Diagnosis OHE' and 'Metabolic Syndrome
#Diagnosis OHE' - too many 1s!
#features_df=features_df.join(target_col)
'''
msno.heatmap(features_df)
#msno.dendrogram(features_df)


corr = features_df.corr()
plt.figure(figsize=(12,10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)])
plt.clf()

symptoms = ['loss of taste or smell-Diagnosis date-Days from Reference',
            'dyspnea-Diagnosis date-Days from Reference',
            'fatigue-Diagnosis date-Days from Reference',
            'diarrhea-Diagnosis date-Days from Reference',
            'fever-Diagnosis date-Days from Reference',
            'headache-Diagnosis date-Days from Reference',
            'cough-Diagnosis date-Days from Reference',
            'pharyngitis-Diagnosis date-Days from Reference',
            'abdominal pain-Diagnosis date-Days from Reference']
symptoms_df = features_df[symptoms]

symptoms_ohe_and_target_df = features_df[['loss of taste or smell DATE_OHE',
                                         'dyspnea DATE_OHE',
                                         'fatigue DATE_OHE',
                                         'diarrhea DATE_OHE',
                                         'fever DATE_OHE',
                                         'headache DATE_OHE',
                                         'cough DATE_OHE',
                                         'pharyngitis DATE_OHE',
                                         'abdominal pain DATE_OHE']].join(target_col)
corr2 = symptoms_ohe_and_target_df.corr()
sns.heatmap(corr2)
'''

features_df.fillna(features_df.mean(), inplace=True)


variable_types = {
                 'Gender': ft.variable_types.Boolean,
                 'Country of birth': ft.variable_types.Categorical,
                 'Pregnancy Weeks': ft.variable_types.Ordinal,
                 'Immigration date': ft.variable_types.Datetime,
                 'Smoking-Smoking status': ft.variable_types.Categorical,
                 'eGFR-Is pregnant': ft.variable_types.Ordinal,
                 'Asthma Diagnosis DATE_OHE': ft.variable_types.Boolean,
                 'COPD Diagnosis DATE_OHE': ft.variable_types.Boolean,
                 'Dementia Diagnosis DATE_OHE': ft.variable_types.Boolean,
                 'Autoimmune Disease Diagnosis DATE_OHE': ft.variable_types.Boolean,
                 'Electrolyte and Fluid Disorder DATE_OHE': ft.variable_types.Boolean,
                 'Depression DATE_OHE': ft.variable_types.Boolean,
                 'Asthma Diagnosis MAC DATE_OHE': ft.variable_types.Boolean,
                 'Iron Deficiency Anemia DATE_OHE': ft.variable_types.Boolean,
                 'Chronic Pulmonary Heart Disease DATE_OHE': ft.variable_types.Boolean,
                 'Rheumatoid Arthritis and Other DATE_OHE': ft.variable_types.Boolean,
                 'loss of taste or smell DATE_OHE': ft.variable_types.Boolean,
                 'dyspnea DATE_OHE': ft.variable_types.Boolean,
                 'fatigue DATE_OHE': ft.variable_types.Boolean,
                 'diarrhea DATE_OHE': ft.variable_types.Boolean,
                 'fever DATE_OHE': ft.variable_types.Boolean,
                 'headache DATE_OHE': ft.variable_types.Boolean,
                 'cough DATE_OHE': ft.variable_types.Boolean,
                 'pharyngitis DATE_OHE': ft.variable_types.Boolean,
                 'abdominal pain DATE_OHE': ft.variable_types.Boolean
                 }



es = ft.EntitySet(id="features")
es.entity_from_dataframe(entity_id='features_df',
                            dataframe=features_df,
                            make_index=True,
                            index='index',
                            variable_types=variable_types)

"""
es.normalize_entity(base_entity_id='features_df',
                    new_entity_id='target_col',
                    index='index',
                    additional_variables=['target_should_hosp'])
"""

feature_matrix_basic, feature_names_basic = ft.dfs(entityset=es,
                                       target_entity='features_df',
                                       max_depth=1,
                                       #n_jobs=-1,
                                       chunk_size=100
                                       )

feature_matrix, feature_names = ft.dfs(entityset=es,
                                       target_entity='features_df',
                                       max_depth=1,
                                       trans_primitives=['and', 'or', 'percentile'],
                                       #verbose=1,
                                       #n_jobs=-1,
                                       chunk_size=100
                                       )
clf = RandomForestClassifier()

#clf.fit(features_df, target_col)
#print(clf.predict(featurs_df))
#sys.exit()


print(cross_val_score(clf, 
                feature_matrix_basic, 
                target_col,
                #scoring='neg_mean_squared_error',
                cv=10))

print(cross_val_score(clf, 
                feature_matrix, 
                target_col,
                #scoring='neg_mean_squared_error',
                cv=10))
'''
[0.83891213 0.79288703 0.77405858 0.86401674 0.87238494 0.87840671
 0.7672956  0.73584906 0.84696017 0.83647799]
'''
