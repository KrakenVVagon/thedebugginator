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

extract = Extractor(training_path)
df = extract.get_df()

df.columns = [x.split('.')[1] for x in df.columns]

remove_cols = ['appid','spaceid','sessionid','playersessionid',
             'profileid','totalplaytime','userid','clientip',
             'countrycode','absoluteplaytime','relativeplaytime',
             'release','app_name','spacename','country','business_region',
             'geo_continent','subcontinent','region','serverdate','p_dateid',
             'createddate','environment','installmentname','issampled','offline',
             'eventid','enemyarchdescription','combatweaponusedname','combattypeofkllname',
             'playerheatlevel','powerlevel','enemylvl','copfelony','crimfelony','factionid'
        ]

cols = [c for c in df.columns if "context" not in c and c not in remove_cols]

df = df[cols]

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
