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
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

LOGDIR = 'logs/sample6/'


# Define a simple convolutional layer
def conv_layer(inputs, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# And a fully connected layer
def fc_layer(inputs, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(inputs, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    # Train for 2000 steps
    loss, acc = 0., 0.
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        # Run the training step
        _, train_loss, train_accuracy = sess.run([train_step, xent, accuracy],
                                                 feed_dict={x: batch[0], y: batch[1]})
        loss += train_loss
        acc += train_accuracy

        if (i + 1) % 50 == 0:
            summary = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(summary, i+1)
            print("step %d, training loss = %g, training accuracy = %g" % (i+1, loss/50., acc/50.))
            loss, acc = 0., 0.

        if (i + 1) % 1000 == 0:
            saver.save(sess, os.path.join(LOGDIR, hparam, "model.ckpt"), i+1)


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-3, 1E-4, 1E-5]:

        # Include "False" as a value to try different model architectures
        for use_two_fc in [True, False]:
            for use_two_conv in [True, False]:
                # Construct a hyperparameter string for each one
                # (example: "lr_1E-3,fc=2,conv=2)
                hparam_str = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam_str)

                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam_str)


if __name__ == '__main__':
    main()
