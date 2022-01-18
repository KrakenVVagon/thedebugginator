#-*- coding: utf-8 -*-
'''
Classes and functions to build an autoencoder

Created by: Andrew Younger
2022-01-18
'''
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector,self).__init__()
        self.encoder = Sequential([

            ])

        self.decoder = Sequential([

            ])

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
