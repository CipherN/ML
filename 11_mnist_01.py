from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST', one_hot=True)

# input_layer
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')

# label
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

batch_size = 1000


def add_layer(input_data, input_num, output_num, activation_function=None):
    # output = input_data * weight + bias
    w = tf.Variable(initial_value=tf.random_normal(shape=[input_num, output_num]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_num]))
    output = tf.add(tf.matmul(input_data, w), b)

    # activation_function ? output = acivation_function(output) : output
    if activation_function:
        output = activation_function(output)

    return output


def build_nn(data):
    hidden_layer1 = add_layer(data, 784, 100,  activation_function=tf.nn.sigmoid)
    hidden_layer2 = add_layer(hidden_layer1, 100, 50,  activation_function=tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2, 50, 10)
    return output_layer


def train_nn(data):
    # output of nn
    output = build_nn(data)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(50):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_data, y_data = mnist.train.next_batch(batch_size)
                cost, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
                epoch_cost += cost
                print('Epoch', i, ':', epoch_cost)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(output, 1)), tf.float32))
        acc = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
        print(acc)


train_nn(x)


