#-*- coding: utf-8 -*-
'''
Classes and functions to extract and preprocess data for debuggination

Created by: Andrew Younger
2022-03-24
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

        predicted = self.predict(self.df)
        print(f'Original dimensions: {self.df.shape}')
        print(f'Encoded dimensions: {predicted.shape}')

        return None

    def predict(self,x,numerical_features=None,categorical_features=None):
        if self.model is None:
            raise AttributeError('No trained model to make predictions. Run Preprocesser.train() first')

        numericals = numerical_feautres or self.numerical_features
        categoricals = categorical_features or self.categorical_features

        predict_list = [x[c] for c in numericals + categoricals]
        return self.model.predict(predict_list)

    def save(self,fname,fpath='/root/thedebugginator/models'):
        if self.model is None:
            raise AttributeError('No existing model to save.')
        self.model.save(f'{fpath}/{fname}')

class Extractor():
    def __new__(self,pyspark=False,ftype=None):
        if pyspark:
            return PySparkExtractor()
        else:
            return PandasExtractor(ftype=ftype)

class PandasExtractor():
    '''
    ftype: optional parameter to specify the type of fstring given (pandas readable). defaults to None
    '''
    def __init__(self,ftype=None):
        self.ftype = ftype
        
    def get_df(self,fstring,feature_cols=[],exclude_cols=[],**kwargs):
        ftype = self.ftype or fstring.split('.')[-1]

        if ftype == 'csv' or ftype == 'txt':
            df = pd.read_csv(fstring,**kwargs)
        elif ftype in ['xlsx','xlsm','xlsb','xls']:
            df = pd.read_excel(fstring,**kwargs)
        elif ftype == 'json':
            df = pd.read_json(fstring,**kwargs)
        else:
            raise ValueError(f'File type {ftype} not currently supported')
        self.df = df 
        if (self._lencheck(feature_cols) or self._lencheck(exclude_cols)):
            self.tdf = self.extract_features(df=df,feature_cols=feature_cols,exclude_cols=exclude_cols)
            return self.tdf
        return df

    def _lencheck(self,x):
        return len(x) > 0

    def extract_features(self,df=None,feature_cols=[],exclude_cols=[],inplace=False):
        if df is None:
            df = self.df.copy()

        use_feature_cols = self._lencheck(feature_cols)
        use_exclude_cols = self._lencheck(exclude_cols)

        if not (use_feature_cols or use_exclude_cols):
            raise ValueError('No feature columns to extract')

        tdf = df.drop(exclude_cols,axis=1)
        if use_feature_cols:
            tdf = tdf[feature_cols]
        self.tdf = tdf

        if inplace==True:
            self.df = self.tdf
            return None
        return tdf

    def save_df(self,df,savepath,**kwargs):
        ftype = savepath.split('.')[-1]

        if ftype == 'csv' or ftype == 'txt':
            df.to_csv(savepath,**kwargs)
        elif ftype in ['xlsx','xlsm','xlsb','xls']:
            df.to_excel(savepath,**kwargs)
        elif ftype == 'json':
            df.to_json(savepath,**kwargs)
        else:
            raise ValueError(f'File type {ftype} not currently supported')
        
        print(f'Saved dataframe to {savepath}')
        return None

    def default_extraction(self,df=None,datapath='/root/thedebugginator/data/raw/default_columns.txt'):
        if df is None:
            df = self.df.copy()
        
        # by default the column name is just the last part after . in the original column names
        df.columns = [x.split('.')[-1] for x in df.columns]

        # remove any columns that have "context" in the name
        no_context_cols = [x for x in df.columns if "context" not in x.lower()]

        # remove list of default columns (list found in the datafiles)
        with open(datapath,'r') as f:
            droplist = f.readlines()
            droplist = [x.split()[0] for x in droplist]
            
        keep_cols = [x for x in no_context_cols if x.lower() not in droplist]

        return df[keep_cols]

# TODO: make this work    
class PySparkExtractor():
    '''
    PySpark extractor class that will extract the data from a pyspark dataframe
    '''
    def __init__(self):
        return None

    def get_df():
        return None

    def _lencheck():
        return None

    def extract_features():
        return None

    def save_df():
        return None

    def _default_extraction():
        return None

def main():
    return True

if __name__ == '__main__':
    main()
