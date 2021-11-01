# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:20:50 2020

@author: hersh.ravkin
"""

import sys
import numpy as np
import pandas as pd
import load_features
import load_target_variable

csv_path = "H:/Manager/datasets/v7.1.csv"

"""
Dataset Generation (calling the above code)
"""
target_col = load_target_variable.load_target_variable(csv_path)
features_df = load_features.load_features(csv_path)
#QA: check out whats going on with 'Asthma Diagnosis OHE' and 'Metabolic Syndrome
#Diagnosis OHE' - too many 1s!