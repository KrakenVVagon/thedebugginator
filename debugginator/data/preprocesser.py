#-*- coding: utf-8 -*-
'''
Classes and functions to encode and normalize debugginator data

Created by: Andrew Younger
2022-01-09
'''
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Preprocesser():
    '''
    Preprocesser class takes a pandas dataframe as input and additional arguments to specify the encoding needed for the column
    Used like a general model except done on a specific dataframe not a general object

    df : pandas dataframe input argument
    categorical_features : optional arugment to specify which columns should be vectorized. defaults to empty list
    numerical_features : optional argument to specify which columns should be normalized. defaults to empty list

    predict method : takes an array of the correct size and perfroms the preprocessing
    train method : uses the input df to train the preprocesser
    '''
    def __init__(self,df,categorical_features=[],numerical_features=[]):
        self.df = df
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.normalized_features = []
        self.numerical_inputs = []
        self.categorical_inputs = []
        self.encoded_features = []
        self.model = None
        
        if len(self.categorical_features) == 0:
            self.categorical_features = [c for c in self.df.columns if self.df[c].dtype not in ['int64','float64']]
        if len(self.numerical_features) == 0:
            self.numerical_features = [c for c in self.df.columns if self.df[c].dtype in ['int64','float64']]

    def normalize(self):
        '''
        Normalizes the numerical features of the dataframe. Resets the normalization attributes to empty.
        '''
        self.normalized_features = []
        self.numerical_inputs = []
        for n in self.numerical_features:
            ninput = layers.Input(shape=(1,),dtype=tf.float32)
            normalizer = layers.Normalization()
            normalizer.adapt(self.df[n])
            normalized_data = normalizer(ninput)
            self.numerical_inputs.append(ninput)
            self.normalized_features.append(normalized_data)

        print(f'Normalized {len(self.numerical_features)} of {len(self.df.columns)} total columns')

    def encode(self):
        '''
        Encodes the categorical features of the dataframe. Resets the encoding attributes to empty
        Currently only supports one-hot encoding of features
        '''
        self.categorical_inputs = []
        self.encoded_features = []
        for c in self.categorical_features:
            cinput = layers.Input(shape=(1,),dtype=tf.string)
            encoder = layers.StringLookup(output_mode='one_hot')
            encoder.adapt(self.df[c].astype(str))
            encoded = encoder(cinput)
            self.categorical_inputs.append(cinput)
            self.encoded_features.append(encoded)

        print(f'Encoded {len(self.categorical_features)} of {len(self.df.columns)} total columns')

    def train(self):
        '''
        Train the preprocesser to accept inputs in the same form as its own df
        Assigns a model to the preprocesser that can be used to predict things
        '''
        # check if the encoded features and/or normalized features exist and are not empty.
        if len(self.encoded_features) == 0:
            self.encode()
        if len(self.normalized_features) == 0:
            self.normalize()

        output = layers.concatenate(self.normalized_features + self.encoded_features)
        self.model = Model(inputs=self.numerical_inputs+self.categorical_inputs,outputs=[output])
        print('Preprocess training finished')

        predict_list = [self.df[c] for c in self.numerical_features + self.categorical_features]
        predicted = self.predict(predict_list)
        print(f'Original dimensions: {self.df.shape}')
        print(f'Encoded dimensions: {predicted.shape}')

        return None

    def predict(self,x):
        if self.model is None:
            raise AttributeError('No trained model to make predictions. Run Preprocesser.train() first')
        return self.model.predict(x)

    def save(self,fname,fpath='/root/thedebugginator/models'):
        if self.model is None:
            raise AttributeError('No existing model to save.')
        self.model.save(f'{fpath}/{fname}')

def main():
    x = pd.DataFrame({
        'id': [1,2,3,4,5],
        'colour': ['red','blue','red','green','red']
        })

    pp = Preprocesser(x)
    pp.train()

if __name__ == '__main__':
    main()
