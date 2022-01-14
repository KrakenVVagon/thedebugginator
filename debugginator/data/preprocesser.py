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
        
        if len(self.categorical_features) == 0:
            self.categorical_features = [c for c in self.df.columns if self.df[c].dtype not in ['int64','float64']]
        if len(self.numerical_features) == 0:
            self.numerical_features = [c for c in self.df.columns if self.df[c].dtype in ['int64','float64']]

    def normalize(self,x):
        return None

    def encode(self,x):
        return None

def main():
    x = pd.DataFrame({
        'id': [1,2,3,4,5],
        'colour': ['red','blue','red','green','red']
        })

    pp = Preprocesser(x)
    print(pp.categorical_features)
    print(pp.numerical_features)

if __name__ == '__main__':
    main()
