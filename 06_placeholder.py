import tensorflow as tf

input1 = tf.placeholder(tf.float32) #placeholder意味着先不给定value，在print时候再通过feed_dict给定相应的输入值

input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [2.], input2: [7.]})) #feed_dict一定以字典形式与placeholder配合使用
