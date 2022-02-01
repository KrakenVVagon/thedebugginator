# -*- coding: utf-8 -*-
"""
Running script for thedebugginator

author: Andrew Younger
2020-01-18
"""

import pandas as pd
import tensorflow as tf
from debugginator.data.extractor import Extractor

file_path = '/root/thedebugginator/data/raw/drone_melee_kills_1000.csv'

extract = Extractor()
df = extract.get_df(file_path)

preprocesser = tf.keras.models.load_model('/root/thedebugginator/models/testing_processer')

df = extract.default_extraction(df=df)

numerical_features = ['ai_positionx','ai_positiony','ai_positionz',
            'playerpositionx','playerpositiony','playerpositionz','killdist']

categorical_features = [c for c in df.columns if c not in numerical_features]
df[categorical_features] = df[categorical_features].astype(str)

prediction_list = [df[c] for c in numerical_features + categorical_features]
x = preprocesser.predict(prediction_list)

print(x)
