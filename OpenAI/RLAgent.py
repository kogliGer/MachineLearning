import tensorflow as tf

gamma = 0.7


x = tf.placeholder(tf.float32, shape=[1, 4])    #state
y = tf.placeholder(tf.float32, shape=[1, 1])    #action

#4 hidden neurons
W = tf.Variable(tf.random_normal([4, 4]))
b = tf.Variable(tf.random_normal([4, 1]))

output = tf.sigmoid(tf.matmul(x, W) + b)


def learn(lastState, actionExecuted, reward, currentState):
    #loss = tf.reduce_sum(((reward + gamma * tf._max())))
    print("learning")
    return 1


def feedForward(state, action):
    with tf.Session() as sess:
        return sess.run([output], feed_dict={x: [state, action]})