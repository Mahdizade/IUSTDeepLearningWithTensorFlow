# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

LOGDIR = 'logs/sample3/'


# Define a simple convolutional layer
def conv_layer(inputs, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]), name="W")
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# And a fully connected layer
def fc_layer(inputs, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in, channels_out]), name="W")
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        return tf.nn.relu(tf.matmul(inputs, w) + b)

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
x_image = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
conv1 = conv_layer(x_image, 1, 32, "conv1")
conv2 = conv_layer(conv1, 32, 64, "conv2")
flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
logits = fc_layer(fc1, 1024, 10, "fc2")

with tf.name_scope("xent"):
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Train for 2000 steps
loss, acc = 0., 0.
for i in range(2000):
    batch = mnist.train.next_batch(50)
    # Run the training step
    _, train_loss, train_accuracy = sess.run([train_step, xent, accuracy],
                                             feed_dict={x: batch[0], y: batch[1]})
    loss += train_loss
    acc += train_accuracy

    if (i+1) % 50 == 0:
        print("step %d, training loss = %g, training accuracy = %g" % (i+1, loss/50., acc/50.))
        loss, acc = 0., 0.
