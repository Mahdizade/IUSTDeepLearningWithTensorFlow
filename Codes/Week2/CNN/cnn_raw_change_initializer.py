"""
Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_epochs = 1
batch_size = 100
log_interval = 10

# Number of Iterations
train_iterations = int(mnist.train.num_examples/batch_size)
validation_iterations = int(mnist.validation.num_examples/batch_size)
test_iterations = int(mnist.test.num_examples/batch_size)

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


# Create some wrappers for simplicity
def conv2d(inputs, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    inputs = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
    inputs = tf.nn.bias_add(inputs, b)
    return tf.nn.relu(inputs)


def maxpool2d(inputs, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, num_classes], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=0.1)),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=0.1)),
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([num_classes], stddev=0.1))
}

# Construct model
# tf Graph input
# MNIST data input is a 1-D vector of 784 features (28*28 pixels)
input_images = tf.placeholder(tf.float32, [None, num_input])
labels = tf.placeholder(tf.float32, [None, num_classes])
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)

# Reshape to match picture format [Height x Width x Channel]
# Tensor input become 4-D:
# [Batch Size, Height, Width, Channel]
inputs_reshaped = tf.reshape(input_images, shape=[-1, 28, 28, 1])

# Convolution Layer
# input_shape = [None, 28, 28, 1]
# output_shape = [None, 28, 28, 32]
conv1 = conv2d(inputs_reshaped, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
# input_shape = [None, 28, 28, 32]
# output_shape = [None, 14, 14, 32]
max_pool1 = maxpool2d(conv1, k=2)

# Convolution Layer
# input_shape = [None, 14, 14, 32]
# output_shape = [None, 14, 14, 64]
conv2 = conv2d(max_pool1, weights['wc2'], biases['bc2'])
# Max Pooling (down-sampling)
# input_shape = [None, 14, 14, 64]
# output_shape = [None, 7, 7, 64]
max_pool2 = maxpool2d(conv2, k=2)

# Fully connected layer
# Reshape max_pool2 output to fit fully connected layer input
# input_shape = [None, 7, 7, 64]
# output_shape = [None, 3136]
fc1_input = tf.reshape(max_pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.nn.relu(tf.nn.xw_plus_b(fc1_input, weights['wd1'], biases['bd1']))
# Apply Dropout
after_dropout = tf.nn.dropout(fc1, keep_prob)

# Output, class prediction
logits = tf.nn.xw_plus_b(after_dropout, weights['out'], biases['out'])

prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init_op = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init_op)

    for epoch in range(num_epochs):
        # TRAIN
        loss, acc = 0., 0.
        for iteration in range(train_iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            _, batch_loss, batch_acc = sess.run([train_op, loss_op, accuracy], feed_dict={
                input_images: batch_x, labels: batch_y, keep_prob: 0.75})
            loss += batch_loss
            acc += batch_acc
            if (iteration+1) % log_interval == 0:
                print('iteration %d' % (iteration+1))
                print('train loss = %s, train accuracy = %s' % (loss/log_interval, acc/log_interval))
                loss, acc = 0, 0

        # VALIDATION
        loss, acc = 0., 0.
        for iteration in range(validation_iterations):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            batch_loss, batch_acc = sess.run([loss_op, accuracy], feed_dict={
                input_images: batch_x, labels: batch_y, keep_prob: 1.})
            loss += batch_loss
            acc += batch_acc
        print('validation loss = %s, validation accuracy = %s'
              % (loss/validation_iterations, acc/validation_iterations))

    print("Optimization Finished!")

    # TEST
    loss, acc = 0., 0.
    for iteration in range(test_iterations):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        batch_loss, batch_acc = sess.run([loss_op, accuracy], feed_dict={
            input_images: batch_x, labels: batch_y, keep_prob: 1.})
        loss += batch_loss
        acc += batch_acc
    print('test loss = %s, test accuracy = %s' % (loss / test_iterations, acc / test_iterations))
