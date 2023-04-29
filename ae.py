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

class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__(input_dim, latent_dim)

        inputs = tf.keras.Input(shape=(input_dim,), name='encoder_input')
        x = tf.keras.layers.Dense(400, activation='sigmoid', name='encoder_dense0')(inputs)
        x = tf.keras.layers.Dense(100, activation='sigmoid', name='encoder_dense1')(x)
        x = tf.keras.layers.Dense(10, activation='sigmoid', name='encoder_dense2')(x)
        x = tf.keras.layers.Dense(latent_dim, activation='tanh', name='encoder_dense3')(x)
        
        mu = tf.keras.layers.Dense(units=latent_dim, name='encoder_mu')(x)
        var = tf.keras.layers.Dense(units=latent_dim, name='encoder_var')(x)
        sam = tf.keras.layers.Lambda(self.sampling, name='encoder_sampling')([mu, var])

        self.encoder = tf.keras.Model(inputs, [mu, var, sam], name='encoder')

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
        return random_sample

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }