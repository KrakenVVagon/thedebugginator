# -*- coding: utf-8 -*-
"""
Training script for thedebugginator

author: Andrew Younger
2020-01-18
"""
import pandas as pd
import tensorflow as tf
from debugginator.data.extractor import Extractor
from debugginator.data.preprocesser import Preprocesser

training_path = './data/raw/drone_bullet_proper_10k.csv'

extract = Extractor()
df = extract.get_df(training_path)

df = extract.default_extraction(df=df)

numerical_features = ['ai_positionx','ai_positiony','ai_positionz',
                      'playerpositionx','playerpositiony','playerpositionz','killdist'
        ]

categorical_features = [c for c in df.columns if c not in numerical_features]
df[categorical_features] = df[categorical_features].astype(str)

processer = Preprocesser(df)
processer.train()

processer.save('testing_processer')
print('Saved preprocesser')
print('Training completed')
