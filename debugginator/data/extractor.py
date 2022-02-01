#-*- coding: utf-8 -*-
'''
Classes and functions to extract or read data for preprocessing

Created by: Andrew Younger
2022-01-09
'''
import pandas as pd

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

    def save_df(self,savepath,**kwargs):
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
    data_path = '/root/thedebugginator/data/raw/drone_bullet_proper_10k.csv'
    extractor = Extractor()

    df = extractor.get_df(data_path)
    print(df.head(10))

    tdf = extractor.default_extraction()
    print(tdf.head(10))

    tdf = extractor.extract_features(df=df,feature_cols=['fact_playerkill.killdist'],exclude_cols=['fact_playerkill.appid'])
    print(tdf.head(10))

    pysparkExtractor = Extractor(pyspark=True)
    print(pysparkExtractor)

if __name__ == '__main__':
    main()
