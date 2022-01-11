# -*- coding: utf-8 -*-

from dataclasses import dataclass
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText

class EventTable(pd.DataFrame):
    '''
    Used for testing the frequency of an event. Should take some sort of dataframe or dict
    DataFrame should be |UUID|dateid|version|platform|duration|event|count
    '''
    
    @property
    def _constructor(self):
        return EventTable
    
    def detectShocks(self,p,freqcol='frequency',qcol='Q',gcols=[]):
        '''Determine where the daily frequency peaks or valleys. Corresponds to a given Q value (percent increase)'''
        if freqcol not in self.columns:
            raise AttributeError('No frequency column detected')
        
        if qcol not in self.columns:
            self.getQFactor(freqcol,gcols=gcols)
            
        self['shock_detected'] = self[qcol] >= p
        
        return None
    
    def getFrequency(self,gcols,mcols,how='sum'):
        '''Get frequency for a given set of group columns. Options for how to aggregate (default sum) and which columns to measure (default all)'''
        if len(mcols) != 2:
            raise ValueError('Measurement columns should have two values. E.g. ["count","duration"]')
            
        aggdict = {x:how for x in mcols}
        freqdf =  self.groupby(gcols).agg(aggdict)
        freqdf['frequency'] = freqdf[mcols[0]]/freqdf[mcols[1]]
        
        freqdf = freqdf.groupby(gcols[:-1]).agg({'frequency':'mean'})
        
        return EventTable(freqdf)
    
    def getMovingAvg(self,col,n):
        '''Adds a moving average column to the table for a desired window and column'''
        self['MA_{}'.format(str(col))] = self[col].rolling(window=n).mean()
        return None
    
    def getQFactor(self,col,gcols=[]):
        '''Adds a Q factor column to the table'''
        
        if len(gcols) == 0:
            self['Q'] = (np.abs(self[col]-self[col].shift())) / (self[col].shift()+self[col])
        else:
            gdf = self.groupby(gcols)
            self['Q'] = (np.abs(gdf[col].shift(0)-gdf[col].shift())) / (gdf[col].shift()+gdf[col].shift(0))
        return None
    
    def shockNotice(self,sentcol='notice_sent'):
        '''Adds a column for which notices have been sent or not'''
        if sentcol not in self.columns:
            self[sentcol] = False
            
        noNoticeSent = (self['shock_detected']==True) & (self['notice_sent']==False)
        
        for ind in self[noNoticeSent].index:
            message = 'Frequency degradation detected'
            notice = FrequencyDegrade(message=message,degrade_params=ind,Q=self[noNoticeSent].loc[ind]['Q'])
            notice.send_notice()
            
        self.loc[noNoticeSent==True,sentcol] = True
        
        return None
    
def main():
    # read text file that has the correct column/event setup
    df = pd.read_csv('edo_test_data_single_user.txt')
    df = EventTable(df)
    
    x = df.getFrequency(['event','date','uuid'],mcols=['number','duration'])
    x.detectShocks(.5,gcols=['event'])
    x.shockNotice()
    
if __name__ == '__main__':
    main()