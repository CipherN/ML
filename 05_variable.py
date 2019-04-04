import tensorflow as tf

#variable definition
state = tf.Variable(0, name='counter')

print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)    #assign new_value to state

#variable initialize
init = tf.global_variables_initializer()    #it is needed for variable definition.
with tf.Session() as sess:
    sess.run(init)

    for step in range(3):
        sess.run(update)

        print(step, sess.run(state), sess.run(new_value))

