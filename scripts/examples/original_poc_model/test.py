# -*- coding: utf-8 -*-
'''
Example script to read and run a selected model on some data

Created: 2022-03-31
Author: Andrew Younger
'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
import debugginator.data
import json

extractor = debugginator.data.Extractor()
preprocesser = load_model('/root/thedebugginator/models/example_preprocesser')
autoencoder = load_model('/root/thedebugginator/models/example_autoencoder')

test_df = extractor.get_df('/root/thedebugginator/data/raw/drone_bullet_no_weapon_kills_1000.csv')
test_df = extractor.default_extraction(df=test_df)

numerical_features = [
        'ai_positionx','ai_positiony','ai_positionz',
        'playerpositionx','playerpositiony','playerpositionz','killdist'
        ]

categorical_features = [c for c in test_df.columns if c not in numerical_features]

test_df.fillna(value={'combattakedown':0,'takedownstate':0},inplace=True)
test_df[categorical_features] = test_df[categorical_features].astype(str)

bad_columns = [
        'enemyarchdescription','combatweaponusedname','combattypeofkillname',
        'playerheatlevel','powerlevel','enemylvl','copfelony','crimfelony','factionid'
        ]

test_df.drop(bad_columns,inplace=True,axis=1)
feature_list = [test_df[c] for c in numerical_features + categorical_features if c not in bad_columns]

processed_data = preprocesser.predict(feature_list)
reconstructions = autoencoder.predict(processed_data)

ae_stats_path = '/root/thedebugginator/models/example_autoencoder/model_stats.json'
with open(ae_stats_path,'r') as f:
    contents = json.loads(f.read())

threshold = float(contents['Threshold'])
loss = losses.mae(reconstructions,processed_data)
predictions = tf.math.less(loss,threshold).numpy()

bug_events = test_df.iloc[predictions==False]
print(bug_events)
print('In an actual script you would save these bugged events!')
