#-*- coding: utf-8 -*-
'''
Classes and functions to build an autoencoder

Created by: Andrew Younger
2022-01-18
'''
import tensorflow as tf
from math import log2
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model

class Encoder():
    '''
    Encoder should take the initial size and the depth
    '''
    def __new__(cls,decoder=False):
        if decoder==True:
            return Decoder()
        return super(Encoder,cls).__new__(cls)

    def __init__(self):
        return None

    def __call__(self,initial_size,depth=1):

        layer_list = []
        for i in range(depth):
            node_count = 6
            layer_list.append(layers.Dense(node_count,activation='relu'))
        
        return Sequential(layer_list)

class Decoder():
    '''
    Decoder should take the output size (usually equal to the initial size), the condensed size, and the depth
    '''
    def __init__(self):
        return None

    def __call__(self,output_size,depth=1):

        layer_list = []
        for i in range(depth):
            node_count = 6
            if i+1 == depth:
                layer_list.append(layers.Dense(output_size,activation='sigmoid'))
            else:
                layer_list.append(layers.Dense(node_count,activation='relu'))

        return Sequential(layer_list)

class AutoEncoder(Model):
    '''
    Basic class for an anomaly detector using an autoencoder
    Should take the initial dimensions and the depth of learning as parameters
    encoder and decoder parameters should be none or some keras sequential type
    '''
    def __init__(self,initial_size=None,encoder_depth=1,encoder=None,decoder=None):
        super(AutoEncoder,self).__init__()
        self.initial_size = initial_size
        self.encoder_depth = encoder_depth

        if initial_size is None and (encoder is None or decoder is None):
            raise AttributeError('Initial size cannot be None if encoder and decoder are not defined')

        max_depth = int(log2(initial_size))
        if max_depth < self.encoder_depth
            raise ValueError('Encoder depth is more than greatest possible depth')

        self.encoder = Encoder()(self.initial_size,depth=self.encoder_depth) or encoder
        self.decoder = Decoder()(self.initial_size,depth=self.encoder_depth) or decoder

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        return {
                'initial_size':self.initial_size,
                'encoder_depth':self.encoder_depth,
                }

def main():
    a = AutoEncoder(6)
    print(a.get_config())

if __name__ == '__main__':
    main()
