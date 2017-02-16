import tensorflow as tf

gamma = 0.7


x = tf.placeholder(tf.float32, shape=[1, 4])    #state
y = tf.placeholder(tf.float32, shape=[4, 1])    #action

#4 hidden neurons
W = tf.Variable(tf.random_normal([4, 4]))
b = tf.Variable(tf.random_normal([4, 1]))

output = tf.sigmoid(tf.matmul(x, W) + b)
prediction = tf.arg_max(output)
loss = tf.reduce_sum(tf.square(output - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.initialize_all_variables()
def learn(lastState, actionExecuted, reward, currentState):
    with tf.Session() as sess:
        sess.run(init)
                greedyAction, netOutput = sess.run([prediction, output], feed_dict={x: currentState})
                #if np.random.rand(1) < e:
                #    a[0] = env.action_space.sample()

                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1
                # Train our network using target and predicted Q values
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
        return greedyAction


def feedForward(state, action):
    with tf.Session() as sess:
        return sess.run([output], feed_dict={x: [state, action]})