import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

import TrainLog as tl
import AutoEncoder as ae

epochs_count = 100
batch_size = 128

input_dim = 28*28
latent_vec_dim = 2

autoencoder = ae.AutoEncoder(input_dim, latent_vec_dim)

(train_samples, train_responses), (test_samples, test_responses) = mnist.load_data()

train_samples = train_samples.astype('float32') / 255
train_samples = np.reshape(train_samples, (-1, 784))

test_samples = test_samples.astype('float32') / 255
test_samples = np.reshape(test_samples, (-1, 784))

autoencoder.compile(loss='binary_crossentropy', optimizer='adam')


cb = tl.TrainLog(autoencoder, train_samples, train_responses)
# history = autoencoder.fit(train_samples, train_samples, epochs=epochs_count, batch_size=batch_size,
#                           shuffle=True, validation_data=(test_samples, test_samples),
#                           callbacks=[cb])

# plt.clf()
# plt.cla()
# plt.figure(figsize=(6, 6))
# plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Validation')
# plt.ylabel('Cross Entropy Loss')
# plt.xlabel('Epochs')
# plt.title('Learning Loss Curve', pad=13)
# plt.legend(loc='upper right')
# plt.savefig('./out/loss.svg')

cb.pack_gif('./out/dec_map{x}.png', './out/dec_map.gif', epochs_count)
cb.pack_gif('./out/latent{x}.png', './out/latent.gif', epochs_count)
cb.pack_gif('./out/samples{x}.png', './out/samples.gif', epochs_count)

