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

extract = Extractor(file_path)
df = extract.get_df()

preprocesser = tf.keras.models.load_model('/root/thedebugginator/models/testing_processer')

df.columns = [x.split('.')[1] for x in df.columns]

remove_cols = ['appid','spaceid','sessionid','playersessionid',
            'profileid','totalplaytime','userid','clientip',
            'countrycode','absoluteplaytime','relativeplaytime',
            'release','app_name','spacename','country','business_region',
            'geo_continent','subcontinent','region','serverdate','p_dateid',
            'createddate','environment','installmentname','issampled','offline',
            'eventid','enemyarchdescription','combatweaponusedname','combattypeofkllname',
            'playerheatlevel','powerlevel','enemylvl','copfelony','crimfelony','factionid']

cols = [c for c in df.columns if "context" not in c and c not in remove_cols]

df = df[cols]

numerical_features = ['ai_positionx','ai_positiony','ai_positionz',
            'playerpositionx','playerpositiony','playerpositionz','killdist']

categorical_features = [c for c in df.columns if c not in numerical_features]
df[categorical_features] = df[categorical_features].astype(str)

prediction_list = [df[c] for c in numerical_features + categorical_features]
x = preprocesser.predict(prediction_list)

print(x)
