#-*- coding: utf-8 -*-
'''
Classes and functions to extract or read data for preprocessing

Created by: Andrew Younger
2022-01-09
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Extractor():
    '''
    Extractor class takes a file or path or database link and extracts desired feature columns.
    For the lazy people who do not want to parse their own pandas frames.
    
    fstring: path string or db table name to read and extract from
    pyspark: optional parameter to read as pyspark session. defaults to False
    ftype: optional parameter to specify the type of fstring given (pandas readable). defaults to None
    feature_cols: optional parameter to give a list of columns to keep
    exclude_cols: optional parameter to give a list of columns to exclude
    '''
    def __init__(self,fstring,pyspark=False,ftype=None,feature_cols=[],exclude_cols=[]):
        self.fstring = fstring
        self.pyspark = pyspark
        self.ftype = ftype
        self.feature_cols = feature_cols
        self.exclude_cols = exclude_cols
        self.df = None
        
        if self.pyspark:
            # do pyspark loading things that we need
            self.ftype='pyspark'

        if not self.ftype:
            self.ftype == self.fstring.split('.')[-1]

    def get_df(self,**kwargs):
        if self.ftype == 'csv' or self.ftype == 'txt':
            self.df = pd.read_csv(self.fstring,**kwargs)
        elif self.ftype in ['xlsx','xlsm','xlsb','xls']:
            self.df = pd.read_excel(self.fstring,**kwargs)
        elif self.ftype == 'json':
            self.df = pd.read_json(self.fstring,**kwargs)
        else:
            raise ValueError(f'File type {self.ftype} not currently supported')
        return None

    def extract_features(self,feature_cols=[],exclude_cols=[],inplace=False):
        '''
        Extracts takes only the desired columns from the dataframe.
        If inplace==True replaces self.df with the transformed dataframe (for saving the interim dataframe)
        Adds the attribute "tdf" to the Extractor which is the transformed dataframe
        '''
        if not self.df:
            self.get_df()

        use_feature_cols = True
        use_exclude_cols = True
        
        def lencheck(x):
            return len(x) > 0

        def featurecheck(y,x):
            if lencheck(y):
                return True
            y = x
            return lencheck(y)

        use_feature_cols = feature_check(self.feature_cols,feature_cols)
        use_exclude_cols = feature_check(self.exclude_cols,exclude_cols)

        if not (use_feature_cols or use_exclude_cols):
            print('No features to extract')
            return None

        tdf = self.df.drop(exclude_cols)
        if use_feature_cols:
            tdf = tdf[feature_cols]
        self.tdf = tdf

        if inplace==True:
            self.df = self.tdf
            return None
        return tdf

    def save_df(self,savepath,transformed=True,**kwargs):
        ftype = savepath.split('.')[-1]

        if transformed:
            df = self.tdf
        else:
            df = self.df

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
    
class PySparkExtractor(Extractor):
    '''
    PySpark extractor class that will extract the data from a pyspark dataframe
    '''
    def __init__(self,fstring):
        return None
    
def main():
    print('Extractor main loop')

if __name__ == '__main__':
    main()
