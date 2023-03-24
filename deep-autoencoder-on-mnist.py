import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

input_dim = 28*28
latent_vec_dim = 2


def encoderModel():
    input_layer = Input(shape=(input_dim,))
    encoder_layer_1 = Dense(400, activation='sigmoid')(input_layer)
    encoder_layer_2 = Dense(100, activation='sigmoid')(encoder_layer_1)
    encoder_layer_3 = Dense(10, activation='sigmoid')(encoder_layer_2)
    encoder_layer_4 = Dense(latent_vec_dim, activation='tanh')(encoder_layer_3)
    return Model(input_layer, encoder_layer_4, name="encoder")

def decoderModel():
    input_layer = Input(shape=(latent_vec_dim,))
    decoder_layer_1 = Dense(10, activation='sigmoid')(input_layer)
    decoder_layer_2 = Dense(100, activation='sigmoid')(decoder_layer_1)
    decoder_layer_3 = Dense(400, activation='sigmoid')(decoder_layer_2)
    decoder_layer_4 = Dense(input_dim, activation='sigmoid')(decoder_layer_3)
    return Model(input_layer, decoder_layer_4, name="decoder")

input_layer = Input(shape=(input_dim,))
encoder = encoderModel()(input_layer)
decoder = decoderModel()(encoder)
autoencoder = Model(input_layer, decoder, name="autoencoder")

(train_samples, train_responses), (test_samples, test_responses) = mnist.load_data()

train_samples = train_samples.astype('float32') / 255
train_samples = np.reshape(train_samples, (-1, 784))

test_samples = test_samples.astype('float32') / 255
test_samples = np.reshape(test_samples, (-1, 784))

autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

history = autoencoder.fit(train_samples, train_samples, epochs=10, batch_size=128,
                          shuffle=True, validation_data=(test_samples, test_samples))

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.title('Learning Loss Curve', pad=13)
plt.legend(loc='upper right')
plt.savefig('loss.svg')

latent_predictions = encoder(train_samples)

# pl = np.concatenate(latent_predictions, train_responses, axis=1)
pl = tf.concat([latent_predictions, tf.expand_dims(tf.cast(train_responses, dtype=tf.float32), axis=1)], 1)
npl = pl.numpy()

plt.clf()
plt.scatter(npl[:,0],npl[:,1],s=20, c=npl[:,2], cmap='tab10')
plt.colorbar()
plt.savefig('scatter.svg')
savetxt('pl.csv', pl)

 encoder.save_weights('./ae/ae_enc_400_100_10_2')
decoder.save_weights('./ae/ae_dec_400_100_10_2')

