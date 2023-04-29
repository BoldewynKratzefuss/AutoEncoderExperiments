import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Lambda, Concatenate


img_size = 28
num_channels = 1
latent_space_dim = 2

encoder_main = Sequential([
    Input(shape=(img_size, img_size, num_channels)),
    Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=1),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=1),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=1),
    BatchNormalization(),
    LeakyReLU(),
    Flatten(),
])

encoder_mu = Sequential([
    encoder_main,
    Dense(units=latent_space_dim)
])

encoder_log_variance = Sequential([
    encoder_main,
    Dense(units=latent_space_dim)
])

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

encoder = Sequential([
    Concatenate([encoder_mu, encoder_log_variance]),
    Lambda(sampling)
])

encoder.build(tf.keras.backend.shape([1, 28, 28, 1]))
encoder.summary()
