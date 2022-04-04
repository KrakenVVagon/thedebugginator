#-*- coding: utf-8 -*-
'''
Models for debuggination

Created by: Andrew Younger
2022-03-24
'''
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model

class Encoder(Sequential):
    '''Inherit from Sequential for easier configuration and functionality
        layers_iterable should be list of layers or iterable to create a list of layers from
    '''
    def __init__(self,layers_iterable):
        self.layers_iterable = layers_iterable
        if "keras.layers" in str(type(self.layers_iterable[0])):
            super(Encoder,self).__init__(self.layers_iterable)
        else:
            super(Encoder,self).__init__()
            for k in self.layers_iterable:
                self.add(layers.Dense(k,activation='relu'))

        return None

    def create_decoder(self,max_size,layer_type=layers.Dense,**kwargs):
        '''
        Creates a decoder based on the encoder setup
        max_size is an int for the last step of the decoding process
        '''
        if "keras.layers" in str(type(self.layers_iterable[0])):
            return Decoder(self.layers_iterable[::-1][:-1] + [layer_type(max_size,**kwargs)])
        return Decoder(self.layers_iterable[::-1][:-1] + [max_size])

class Decoder(Sequential):
    '''Decoders are just encoders done backwards'''
    def __init__(self,layers_iterable):
        self.layers_iterable = layers_iterable
        if "keras.layers" in str(type(self.layers_iterable[0])):
            super(Decoder,self).__init__(self.layers_iterable)
        else:
            super(Decoder,self).__init__()
            for k in self.layers_iterable:
                if k == self.layers_iterable[-1]:
                    self.add(layers.Dense(k,activation='sigmoid'))
                    continue
                self.add(layers.Dense(k,activation='relu'))

        return None

    def create_encoder(self,min_size):
        '''
        Creates an encoder from the decoder settings
        min_size represents the latent dimension size for compression
        '''
        if "keras.layers" in str(type(self.layers_iterable[0])):
            return Encoder(self.layers_iterable[::-1][:-1] + [layer_type(min_size,**kwargs)])
        return Encoder(self.layers_iterable[::-1][:-1] + [min_size])

class AutoEncoder(Model):
    '''
    Basic class for an anomaly detector using an autoencoder
    Encoder and decoder should function as keras Sequential objects
    '''
    def __init__(self,encoder,decoder):
        super(AutoEncoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_depth = len(self.encoder.get_config()['layers'])
        self.decoder_depth = len(self.decoder.get_config()['layers'])

        self.encoder_nodes = list(layer['config']['units'] for layer in self.encoder.get_config()['layers'])
        self.decoder_nodes = list(layer['config']['units'] for layer in self.decoder.get_config()['layers'])

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self,verbose=False):
        if verbose:
            return {**self.get_config(),**{
                'encoder_config':self.encoder.get_config(),
                'decoder_config':self.decoder.get_config()
                }}
        return {
                'encoder_depth':self.encoder_depth,
                'encoder_nodes':self.encoder_nodes,
                'deocder_depth':self.decoder_depth,
                'decoder_nodes':self.decoder_nodes
                }

def main():
    l = [10,5,2]
    e = Encoder(l)
    d = Decoder(l[::-1])
    a = AutoEncoder(e,d)
    print(a.decoder.get_config())

if __name__ == '__main__':
    main()
