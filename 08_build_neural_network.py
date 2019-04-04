import tensorflow as tf
import numpy as np

# Build hidden layer
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:     #None表示线性关系
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 5)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# Define placelder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# Add hiden layer
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_function=None)

# Error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - y_data),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Variables initialize
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# Training
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

# Gain the improvement
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}),
                  sess.run(prediction, feed_dict={xs: x_data, ys: y_data}))
