#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pydsutils.generic import create_logger

logger = create_logger(__name__)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding="SAME")


def avg_pool_2x2(x):
    """Average pooling
    
    """
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def discriminator(x_image, reuse=False):
    with tf.variable_scope("discriminator") as _:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # First Conv and Pool Layers
        W_conv1 = tf.get_variable("d_wconv1", [5, 5, 1, 8], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable("d_bconv1", [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)

        # Second Conv and Pool Layers
        W_conv2 = tf.get_variable("d_wconv2", [5, 5, 8, 16], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable("d_bconv2", [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

        # First Fully Connected Layer
        W_fc1 = tf.get_variable("d_wfc1", [7 * 7 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable("d_bfc1", [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Second Fully Connected Layer
        W_fc2 = tf.get_variable("d_wfc2", [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable("d_bfc2", [1], initializer=tf.constant_initializer(0))

        # Final Layer
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def generator(z, batch_size, g_dim=64, c_dim=1, s=28, reuse=False):
    """
    
    :param g_dim: Number of filters of first layer of generator 
    :param c_dim: Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
    :param s: Output size of the image
    """
    with tf.variable_scope("generator") as _:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # We want to slowly upscale the image, so the below values will help make that change gradual
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) 

        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])  # Dimensions of h0 = batch_size x 2 x 2 x 25
        h0 = tf.nn.relu(h0) 

        # First DeConv Layer
        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable("g_wconv1", [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable("g_bconv1", [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], 
                                         padding="SAME")
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, 
                                               scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)
        #Dimensions of H_conv1 = batch_size x 3 x 3 x 256

        # Second DeConv Layer
        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]
        W_conv2 = tf.get_variable("g_wconv2", [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable("g_bconv2", [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], 
                                         padding="SAME")
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, 
                                               scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)
        # Dimensions of H_conv2 = batch_size x 6 x 6 x 128

        # Third DeConv Layer
        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable("g_wconv3", [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable("g_bconv3", [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], 
                                         padding="SAME")
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, 
                                               scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        #Dimensions of H_conv3 = batch_size x 12 x 12 x 64

        #Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable("g_wconv4", [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable("g_bconv4", [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], 
                                         padding="VALID")
        H_conv4 = tf.nn.tanh(H_conv4)  # Dimensions of H_conv4 = batch_size x 28 x 28 x 1
    return H_conv4


##################################################################
train_size = 55000
g_dimensions = 100
logger.info("Program started")

# Data
mnist = input_data.read_data_sets("MNIST_data/")
logger.info("Successfully loaded raw MNIST data")

x_train = mnist.train.images[:train_size,:] 
logger.info("Shapre of x_train: %s " % str(x_train.shape))
image = x_train[random.randint(0, train_size)].reshape([28, 28])
plt.imshow(image, cmap=plt.get_cmap("gray_r"))
plt.show()

# Generator is untrained, so expect a bad result
sess = tf.Session()
g_test_placeholder = tf.placeholder(tf.float32, [None, g_dimensions])
test_z = np.random.normal(-1, 1, [1, g_dimensions])
sample_image = generator(g_test_placeholder, 1, g_dimensions)

sess.run(tf.global_variables_initializer())
temp = sess.run(sample_image, feed_dict={g_test_placeholder: test_z})
my_i = temp.squeeze()  # view the output image
plt.imshow(my_i, cmap='gray_r')
plt.show()

########################################
########################################
batch_size = 16
tf.reset_default_graph()  # Since we changed our batch size (from 1 to 16), need to reset TF graph
sess = tf.Session()
d_placeholder = tf.placeholder("float", shape = [None, 28, 28, 1])  # Placeholder for input images to discriminator
g_placeholder = tf.placeholder(tf.float32, [None, g_dimensions])  # Placeholder for input noise vectors to generator

# Dx will hold discriminator outputs (unnormalized) for the real MNIST images
# Gz holds the generated images
# Dg will hold discriminator outputs (unnormalized) for generated images
Dx = discriminator(d_placeholder) 
Gz = generator(g_placeholder, batch_size, g_dimensions) 
Dg = discriminator(Gz, reuse=True)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if "d_" in var.name]
g_vars = [var for var in tvars if "g_" in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    d_train_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
    g_train_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

sess.run(tf.global_variables_initializer())
iterations = 5000
for iteration in range(iterations):
    g_batch = np.random.normal(-1, 1, size=[batch_size, g_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)
    real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
    # Update the discriminator and generator
    _, _ = sess.run([d_train_optimizer, d_loss], feed_dict={g_placeholder: g_batch,
                        d_placeholder: real_image_batch}) 
    _, _ = sess.run([g_train_optimizer, g_loss], feed_dict={g_placeholder:g_batch})  # Update the generator
    if iteration % 10 == 0:
        logger.info("Successfully completed iteration %5d out of %d" %(iteration, iterations - 1))

# A sample image looks like after training.
sample_image = generator(g_placeholder, 1, g_dimensions, reuse=True)
g_batch = np.random.normal(-1, 1, size=[1, g_dimensions])
temp = sess.run(sample_image, feed_dict={g_placeholder: g_batch})
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')

# Finalize
sess.close()
logger.info("ALL DONE!\n")
