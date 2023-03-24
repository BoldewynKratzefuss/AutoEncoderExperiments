import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

input_dim = 28*28
latent_vec_dim = 2

input_layer = Input(shape=(input_dim,))
encoder_layer_1 = Dense(400, activation='sigmoid')(input_layer)
encoder_layer_2 = Dense(100, activation='sigmoid')(encoder_layer_1)
encoder_layer_3 = Dense(10, activation='sigmoid')(encoder_layer_2)
encoder_layer_4 = Dense(latent_vec_dim, activation='tanh')(encoder_layer_3)
decoder_layer_1 = Dense(10, activation='sigmoid')(encoder_layer_4)
decoder_layer_2 = Dense(100, activation='sigmoid')(decoder_layer_1)
decoder_layer_3 = Dense(400, activation='sigmoid')(decoder_layer_2)
decoder_layer_4 = Dense(input_dim, activation='sigmoid')(decoder_layer_3)

autoencoder = Model(input_layer, decoder_layer_4, name='autoencoder')
encoder = Model(input_layer, encoder_layer_4, name='encoder')

latent_input = Input(shape=(latent_vec_dim,))
decoder = Model(latent_input, autoencoder.layers[-4](latent_input), name='decoder')

(train_samples, train_responses), (test_samples, test_responses) = mnist.load_data()

train_samples = train_samples.astype('float32') / 255
train_samples = np.reshape(train_samples, (-1, 784))

test_samples = test_samples.astype('float32') / 255
test_samples = np.reshape(test_samples, (-1, 784))

autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

history = autoencoder.fit(train_samples, train_samples, epochs=100, batch_size=128,
                          shuffle=True, validation_data=(test_samples, test_samples))

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.title('Learning Loss Curve', pad=13)
plt.legend(loc='upper right')
plt.savefig('loss.svg')

latent_predictions = encoder(train_samples)

pl = tf.concat([latent_predictions, tf.expand_dims(tf.cast(train_responses, dtype=tf.float32), axis=1)], 1)
npl = pl.numpy()

plt.clf()
plt.scatter(npl[:,0],npl[:,1],s=20, c=npl[:,2], cmap='tab10')
plt.colorbar()
plt.savefig('scatter.svg')
savetxt('pl.csv', pl)

encoder.save_weights('./ae/ae_enc_400_100_10_2')
decoder.save_weights('./ae/ae_dec_400_100_10_2')

test_imgs = autoencoder.predict(test_samples)


n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_samples[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(test_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('samples.png')