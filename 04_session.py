import tensorflow as tf

matrix1 = tf.constant([[[3,3]]])

matrix2 = tf.constant([[[2],
                        [2]]])

product = tf.matmul(matrix1, matrix2)   #matmul = matrix multiply 相当于numpy中的np.dot(matrix1, matrix2)

# Method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


# Methood 2
with tf.Session() as sess:  #包含了Method1中sess.run()以及sess.close()
    result2 = sess.run(product)
    print(result2)

