#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://www.kdnuggets.com/2017/10/tensorflow-building-feed-forward-neural-networks-step-by-step.html/3
"""
import tensorflow
import numpy

# Preparing training data (inputs-outputs)
training_inputs = tensorflow.placeholder(shape=[None, 2], dtype=tensorflow.float32)
training_outputs = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32)

"""
Hidden layer with two neurons
"""
# Preparing neural network parameters (weights and bias) using TensorFlow Variables
weights_hidden = tensorflow.Variable(tensorflow.truncated_normal(shape=[2, 2], dtype=tensorflow.float32))
bias_hidden = tensorflow.Variable(tensorflow.truncated_normal(shape=[1, 2], dtype=tensorflow.float32))

# Preparing inputs of the activation function
af_input_hidden = tensorflow.matmul(training_inputs, weights_hidden) + bias_hidden
hidden_layer_output = tensorflow.nn.sigmoid(af_input_hidden)

"""
Output layer with one neuron
"""
# Preparing neural network parameters (weights and bias) using TensorFlow Variables
weights_output = tensorflow.Variable(tensorflow.truncated_normal(shape=[2, 1], dtype=tensorflow.float32))
bias_output = tensorflow.Variable(tensorflow.truncated_normal(shape=[1, 1], dtype=tensorflow.float32))

# Preparing inputs of the activation function
af_input_output = tensorflow.matmul(hidden_layer_output, weights_output) + bias_output
# Activation function of the output layer neuron
predictions = tensorflow.nn.sigmoid(af_input_output)


# Measuring the prediction error of the network after being trained
prediction_error = 0.5 * tensorflow.reduce_sum(tensorflow.subtract(predictions, training_outputs) * tensorflow.subtract(predictions, training_inputs))

# Minimizing the prediction error using gradient descent optimizer
train_op = tensorflow.train.GradientDescentOptimizer(0.05).minimize(prediction_error)

sess = tensorflow.Session()
sess.run(tensorflow.global_variables_initializer())

# Training data inputs
training_inputs_data = [[1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                        [0.0, 0.0]]

# Training data desired outputs
training_outputs_data = [[1.0],
                         [1.0],
                         [0.0],
                         [0.0]]

# Training loop of the neural network
for step in range(10000):
    op, err, p = sess.run(fetches=[train_op, prediction_error, predictions],
                          feed_dict={training_inputs: training_inputs_data,
                                     training_outputs: training_outputs_data})
    print(str(step), ": ", err)

print("Expected class scores : ", sess.run(predictions, feed_dict={training_inputs: training_inputs_data}))

print("Hidden layer initial weights: ", sess.run(weights_hidden))
print("Hidden layer initial bias: ", sess.run(bias_hidden))

print("Output layer weights: ", sess.run(weights_output))
print("Output layer bias: ", sess.run(bias_output))

sess.close()