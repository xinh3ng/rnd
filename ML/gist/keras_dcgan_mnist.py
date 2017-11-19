#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
"""
from pdb import set_trace as debug
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from pydsutils.generic import create_logger

logger = create_logger(__name__)


class DCGAN(object):
    def __init__(self, img_rows, img_cols, channel):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.d = None   # discriminator
        self.g = None   # generator
        self.adv_model = None  # adversarial model
        self.d_model = None  # discriminator model
    
    def get_discrimonator(self):
        return self.d
    def get_generatorself):
        return self.g

    def gen_discriminator(self):
        """
        (W−F+2P)/S+1
        """
        if self.d:
            return self.d
        
        self.d = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth = 64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.d.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.d.add(LeakyReLU(alpha=0.2))
        self.d.add(Dropout(dropout))

        self.d.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.d.add(LeakyReLU(alpha=0.2))
        self.d.add(Dropout(dropout))

        self.d.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.d.add(LeakyReLU(alpha=0.2))
        self.d.add(Dropout(dropout))

        self.d.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.d.add(LeakyReLU(alpha=0.2))
        self.d.add(Dropout(dropout))

        # Out: 1-dim probability
        self.d.add(Flatten())
        self.d.add(Dense(1))
        self.d.add(Activation('sigmoid'))
        self.d.summary()
        return self.d
    
    def gen_generator(self):
        if self.g:
            return self.g
        
        self.g = Sequential()
        dropout = 0.4
        depth = 64 + 64 + 64 + 64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.g.add(Dense(dim*dim*depth, input_dim=100))
        self.g.add(BatchNormalization(momentum=0.9))
        self.g.add(Activation('relu'))
        self.g.add(Reshape((dim, dim, depth)))
        self.g.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.g.add(UpSampling2D())
        self.g.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.g.add(BatchNormalization(momentum=0.9))
        self.g.add(Activation('relu'))

        self.g.add(UpSampling2D())
        self.g.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.g.add(BatchNormalization(momentum=0.9))
        self.g.add(Activation('relu'))

        self.g.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.g.add(BatchNormalization(momentum=0.9))
        self.g.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0, 1.0] per pix
        self.g.add(Conv2DTranspose(1, 5, padding='same'))
        self.g.add(Activation('sigmoid'))
        self.g.summary()
        return self.g
    
    def discriminator_model(self):
        if self.d_model:
            return self.d_model
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.d_model = Sequential()
        self.d_model.add(self.gen_discriminator())
        self.d_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.d_model

    def adversarial_model(self):
        if self.adv_model:
            return self.adv_model
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adv_model = Sequential()
        self.adv_model.add(self.gen_generator())
        self.adv_model.add(self.gen_discriminator())
        self.adv_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.adv_model


class DCGAN_Trainer(object):
    def __init__(self, img_rows, img_cols, channel, 
                 train_steps, batch_size, save_interval):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.save_interval = save_interval

        self.dcgan = DCGAN(self.img_rows, self.img_cols, self.channel)
        self.discriminator = self.dcgan.discriminator_model()
        self.adversarial = self.dcgan.adversarial_model()
        self.generator = self.dcgan.gen_generator()

    def train(self, x_train):
        noise_input = None
        if self.save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(self.train_steps):
            images_train = x_train[np.random.randint(0,
                x_train.shape[0], size=self.batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*self.batch_size, 1])
            y[self.batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([self.batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            logger.info(log_mesg)
            if self.save_interval>0:
                if (i+1) % self.save_interval==0:
                    self.plot_images(x_train, save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, x_train, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, x_train.shape[0], samples)
            images = x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

#########################################
if __name__ == '__main__':
    img_rows = 28
    img_cols = 28
    channel = 1
    train_steps = 2000
    batch_size = 256
    save_interval = 0
                        
    x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
    x_train = x_train.reshape(-1, img_rows, img_cols, 1).astype(np.float32)

    trainer = DCGAN_Trainer(img_rows, img_cols, channel, 
                            train_steps=train_steps, 
                            batch_size=batch_size, 
                            save_interval=save_interval)
    trainer.train(x_train)
    trainer.plot_images(x_train, fake=True)
    trainer.plot_images(x_train, fake=False, save2file=True)
    
    logger.info('ALL DONE!\n')
