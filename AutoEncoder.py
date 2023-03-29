import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
        
class AutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(400, activation='sigmoid'),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(latent_dim, activation='tanh')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(400, activation='sigmoid'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
