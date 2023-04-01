import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
import AutoEncoder

class TrainLog(tf.keras.callbacks.Callback):
    def __init__(self, autoencoder, train_samples, train_responses):
        np_config.enable_numpy_behavior()
        
        self.loop = 0
        self.train_samples = train_samples
        self.train_responses = train_responses
        self.autoencoder = autoencoder
        pass

    def map_latent(self):
        latent = self.autoencoder.encode(self.train_samples)

        pl = tf.concat([latent, tf.expand_dims(tf.cast(self.train_responses, dtype=tf.float32), axis=1)], 1)
        npl = pl.numpy()

        plt.clf()
        plt.cla()
        plt.figure(figsize=(5, 5))
        plt.scatter(npl[:,0],npl[:,1],s=20, c=npl[:,2], cmap='tab10')
        plt.colorbar()
        plt.savefig("./out/latent{x}.svg".format(x=self.loop))
        plt.savefig("./out/latent{x}.png".format(x=self.loop), dpi=96)

    def predict_examples(self):
        test_imgs = self.autoencoder.predict(self.train_samples)

        n = 20  # How many digits we will display
        img_size=0.25
        plt.clf()
        plt.cla()
        plt.figure(figsize=(n*img_size, 2*img_size))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.train_samples[i].reshape(28, 28), cmap='GnBu')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(test_imgs[i].reshape(28, 28), cmap='GnBu')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
        plt.tight_layout(pad=0.0)

        plt.savefig("./out/samples{x}.png".format(x=self.loop), dpi=96)

    def map_encoder(self):
        n = 20  # How many digits we will display
        img_size=0.25

        r = np.linspace(-1, 1, n)
        l = np.meshgrid(r, r)
        l = np.reshape(l, [n*n, 2])

        imgs = self.autoencoder.decode(l)
        plt.clf()
        plt.cla()
        plt.figure(figsize=(n*img_size, n*img_size))
        x = 0
        for r in range(n):
            for c in range(n):
                ax = plt.subplot(n, n, x + 1)
                plt.imshow(imgs[x].reshape(28, 28), cmap='GnBu')
                #plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                x = x+1

        plt.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)
        plt.tight_layout(pad=0.0)
        plt.savefig("./out/dec_map{x}.png".format(x=self.loop), dpi=96)
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.map_latent()
        self.predict_examples()
        self.map_encoder()
        self.loop = self.loop+1

    def pack_gif(self, pattern_in_x, out, count):
        images = []
        frame_length = 0.5 # seconds between frames
        end_pause = 4 # seconds to stay on last frame
        # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
        for ii in range(0, count):       
            images.append(imageio.imread(pattern_in_x.format(x=ii)))
        # the duration is the time spent on each image (1/duration is frame rate)
        imageio.mimsave(out, images,'GIF',duration=frame_length)