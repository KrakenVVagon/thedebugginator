#-*- coding: utf-8 -*-
'''
Classes and functions to build an autoencoder

Created by: Andrew Younger
2022-01-18
'''
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model

class Encoder():
    '''
    Encoder should take the initial size and the depth

    TODO: add logic to get good condensed size. This is all the heavy lifting for the autoencoder
    '''
    def __init__(self,initial_size,depth=1):
        self.output_size=7
        return None

class Decoder():
    '''
    Decoder should take the output size (usually equal to the initial size), the condensed size, and the depth
    '''
    def __init__(self,output_size,encoded_size,depth=1):
        return None

class AutoEncoder(Model):
    '''
    Basic class for an anomaly detector using an autoencoder
    Should take the initial dimensions and the depth of learning as parameters
    '''
    def __init__(self,initial_size,encoder_depth=1):
        super(AutoEncoder,self).__init__()
        self.initial_size = initial_size
        self.encoder_depth = encoder_depth

        self.encoder = Encoder(self.initial_size,depth=self.encoder_depth)
        self.latent_size = self.encoder.output_size
        self.decoder = Decoder(self.initial_size,self.latent_size,depth=self.encoder_depth)

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        return {
                'initial_size':self.initial_size,
                'encoder_depth':self.encoder_depth,
                'latent_size':self.latent_size
                }

def main():
    a = AutoEncoder(6)
    print(a.get_config())

if __name__ == '__main__':
    main()
