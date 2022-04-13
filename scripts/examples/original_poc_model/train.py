# -*- coding: utf-8 -*-
'''
Example script to make the original proof of concept used in the presentation given to stakeholders.

Directly based off the notebook: /thedebugginator/notebooks/debugginator_full_split_testing.pynb

Created: 2022-03-21
Author: Andrew Younger
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
import debugginator.data
import debugginator.models
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

data_path = '/root/thedebugginator/data/raw/drone_bullet_proper_10k.csv'
bug_data  = '/root/thedebugginator/data/raw/drone_bullet_no_weapon_kills_1000.csv'

extractor = debugginator.data.Extractor()

bug_df = extractor.get_df(bug_data)
bug_df['label'] = 0
raw_df = extractor.get_df(data_path)
raw_df['label'] = 1

df = pd.concat([raw_df,bug_df])
df = extractor.default_extraction(df=df)

numerical_features = [
        'ai_positionx','ai_positiony','ai_positionz',
        'playerpositionx','playerpositiony','playerpositionz','killdist'
        ]

categorical_features = [c for c in df.columns if c not in numerical_features and c != 'label']

# categorical features must become strings
df.fillna(value={'combattakedown':0,'takedownstate':0},inplace=True)
df[categorical_features] = df[categorical_features].astype(str)

# remove the columns that are useless or repeated
bad_columns = ['enemyarchdescription','combatweaponusedname','combattypeofkillname',
               'playerheatlevel','powerlevel','enemylvl','copfelony','crimfelony','factionid'
        ]

df.drop(bad_columns,inplace=True,axis=1)

labels = df.pop('label')

# split data into training and testing data
train_data, test_data, train_labels, test_labels = train_test_split(df,labels,test_size=0.2,random_state=7)

# in this script we save some interim things as examples of what future data sets would look like
interim_path = '/root/thedebugginator/data/interim'

interim_savers = [train_data,train_labels]
interim_names  = ['example_training_data','example_training_labels']

for i,df in enumerate(interim_savers):
    extractor.save_df(df=df,savepath=f'{interim_path}/{interim_names[i]}.csv')

preprocesser = debugginator.data.Preprocesser(train_data)
preprocesser.train()
processed_train = preprocesser.predict(train_data)

# save the preprocesser as an example of a preprocesser and the preprocessed data
processed_path = '/root/thedebugginator/data/processed'
extractor.save_df(df=pd.DataFrame(processed_train),savepath=f'{processed_path}/example_processed_data.csv')

preprocesser.save('example_preprocesser')

encoder = debugginator.models.Encoder([32,16,4])
decoder = debugginator.models.Decoder([16,32,processed_train.shape[1]])

autoencoder = debugginator.models.AutoEncoder(encoder=encoder,decoder=decoder)

autoencoder.compile(optimizer='adam',loss='mae')

processed_test = preprocesser.predict(test_data)

history = autoencoder.fit(
        processed_train,processed_train,
        epochs=20,
        batch_size=128,
        validation_data=(processed_test,processed_test),
        shuffle=True
        )

autoencoder.save('/root/thedebugginator/models/example_autoencoder')
print('Saved autoencoder model')

reconstructions = autoencoder.predict(processed_train)
train_loss = losses.mae(reconstructions,processed_train)

def predict(model,data,threshold):
    reconstructions = model(data)
    loss = losses.mae(reconstructions,data)
    return tf.math.less(loss,threshold)

def print_stats(predictions,labels):
    accuracy = accuracy_score(labels,predictions)
    precision= precision_score(labels,predictions)
    recall = recall_score(labels,predictions)
    f1_score = 2*precision*recall/(precision+recall)
    return accuracy, precision, recall, f1_score

threshold = np.mean(train_loss) + np.std(train_loss)
print(f'Testing using 1 STD threshold of: {threshold}')

predictions = predict(autoencoder,processed_test,threshold)
a,p,r,f1 = print_stats(predictions,test_labels)
print(f'Accuracy: {a}')
print(f'Precision: {p}')
print(f'Recall: {r}')
print(f'F1 Score: {f1}')

def save_stats(stats_dict,fname):
    with open(fname,'w') as f:
        json.dump(stats_dict,f,indent=4)

stats_dict = {'Accuracy':str(a),
        'Precision':str(p),
        'Recall':str(r),
        'F1 Score':str(f1),
        'Threshold': str(threshold)
        }

save_stats(stats_dict,'/root/thedebugginator/models/example_autoencoder/model_stats.json')
print('Saved model stats')

# loss plot
plt.figsize=((10,15))
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title('Original POC Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.savefig("/root/thedebugginator/reports/figures/original_poc_training_val_loss.png")
plt.clf()
print('Saved training-validation loss plot')

# plot regular events' reconstruction
plt.figsize=((10,15))
regular_events = train_loss.numpy()[train_labels==1]
plt.hist(regular_events,bins=25)
plt.title('Original POC Regular Reconstruction Error')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.axvline(x=threshold,color='black',linestyle='--')
plt.savefig("/root/thedebugginator/reports/figures/original_poc_regular_reconstruction.png")
plt.clf()
print("Saved regular events' reconstruction plot")

# plot bug events' reconstruction
plt.figsize=((10,15))
bug_events = train_loss.numpy()[train_labels==0]
plt.hist(bug_events,bins=25)
plt.title('Original POC Anomaly Reconstruction Error')
plt.xlabel('Reconstruction Error')
plt.ylabel("Count")
plt.axvline(x=threshold,color='black',linestyle='--')
plt.savefig("/root/thedebugginator/reports/figures/original_poc_anomaly_reconstruction.png")
plt.clf()
print("Saved bug events' reconstruction plot")

# plot bug+regular together
plt.figsize=((10,15))
plt.hist(regular_events,bins=25,label='Regular Events')
plt.hist(bug_events,bins=25,label='Anomalous Events')
plt.title('Original POC Training Reconstruction')
plt.xlabel('Reconstruction Label')
plt.ylabel('Count')
plt.axvline(x=threshold,color='black',linestyle='--')
plt.legend()
plt.savefig("/root/thedebugginator/reports/figures/original_poc_training_reconstruction.png")
plt.clf()
print('Saved training reconstruction plot')

# plot testing reconstruction
plt.figsize=((10,15))
test_reconstruction = autoencoder.predict(processed_test)
test_loss = losses.mae(test_reconstruction,processed_test)
plt.hist(test_loss[None,:],bins=25)
plt.title('Original POC Testing Reconstruction Error')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.axvline(x=threshold,color='black',linestyle='--')
plt.savefig("/root/thedebugginator/reports/figures/original_poc_test_reconstruction.png")
plt.clf()
print('Saved testing reconstruction plot')

print('Training complete')
