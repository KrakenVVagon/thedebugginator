#-*- coding: utf-8 -*-
'''
Classes and functions to extract or read data for preprocessing

Created by: Andrew Younger
2022-01-09
'''
import pandas as pd

class Extractor():
    '''
    Called class that determines for the user if it is a pyspark dataframe or not
    '''
    def __new__(self,fstring,pyspark=False,ftype=None,feature_cols=[],exclude_cols=[]):
        if pyspark:
            return PySparkExtractor(fstring)
        else:
            return PandasExtractor(fstring,ftype=ftype,feature_cols=feature_cols,exclude_cols=exclude_cols)

class PandasExtractor():
    '''
    Extractor class takes a file or path or database link and extracts desired feature columns.
    For the lazy people who do not want to parse their own pandas frames.
    
    fstring: path string or db table name to read and extract from
    ftype: optional parameter to specify the type of fstring given (pandas readable). defaults to None
    feature_cols: optional parameter to give a list of columns to keep
    exclude_cols: optional parameter to give a list of columns to exclude
    '''
    def __init__(self,fstring,ftype=None,feature_cols=[],exclude_cols=[]):
        self.fstring = fstring
        self.ftype = ftype or self.fstring.split('.')[-1]
        self.feature_cols = feature_cols
        self.exclude_cols = exclude_cols
        self.df = None
        
    def get_df(self,**kwargs):
        if self.ftype == 'csv' or self.ftype == 'txt':
            self.df = pd.read_csv(self.fstring,**kwargs)
        elif self.ftype in ['xlsx','xlsm','xlsb','xls']:
            self.df = pd.read_excel(self.fstring,**kwargs)
        elif self.ftype == 'json':
            self.df = pd.read_json(self.fstring,**kwargs)
        else:
            raise ValueError(f'File type {self.ftype} not currently supported')
        return self.df

    def extract_features(self,feature_cols=[],exclude_cols=[],inplace=False):
        '''
        Extracts takes only the desired columns from the dataframe.
        If inplace==True replaces self.df with the transformed dataframe (for saving the interim dataframe)
        Adds the attribute "tdf" to the Extractor which is the transformed dataframe
        '''
        if self.df is None:
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

        use_feature_cols = featurecheck(self.feature_cols,feature_cols)
        use_exclude_cols = featurecheck(self.exclude_cols,exclude_cols)

        if not (use_feature_cols or use_exclude_cols):
            raise ValueError('No feature columns to extract')

        tdf = self.df.drop(self.exclude_cols,axis=1)
        if use_feature_cols:
            tdf = tdf[self.feature_cols]
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
# TODO: make this work    
class PySparkExtractor():
    '''
    PySpark extractor class that will extract the data from a pyspark dataframe
    '''
    def __init__(self,fstring):
        return None
    
def main():
    data_path = '/root/thedebugginator/data/raw/drone_bullet_proper_10k.csv'
    extractor = Extractor(data_path,feature_cols=['fact_playerkill.killdist','fact_playerkill.eventid'])

    tdf = extractor.extract_features()
    print(tdf.head(10))

    pysparkExtractor = Extractor(data_path,pyspark=True)
    print(pysparkExtractor)
if __name__ == '__main__':
    main()
