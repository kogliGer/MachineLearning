import tensorflow as tf
import gym
import numpy as np

gamma = 0.7
epsilon = 0.5
iterations = 10000

x = tf.placeholder(tf.float32, shape=[None, 4])    #state
y = tf.placeholder(tf.float32, shape=[None, 2])    #action

#4 hidden neurons
W = tf.Variable(tf.random_normal([4, 2]))
b = tf.Variable(tf.random_normal([2]))

output = tf.transpose(tf.sigmoid(tf.matmul(x, W) + b))
loss = tf.reduce_sum(tf.square(output - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()

#setup environment
environment = gym.make("CartPole-v0")
newState = np.empty(shape=(1, 4), dtype='float32')
lastState = [[]]
newState[0] = environment.reset()

with tf.Session() as sess:
    for iteration in range(iterations):
        print("iteration " + str(iteration))
        sess.run(init)

        netOutput = sess.run(output, feed_dict={x: newState})
        if np.random.rand(1) < epsilon:
            action = environment.action_space.sample()
        else:

            if(netOutput[0] > netOutput[1]):
                action = 0
            else:
                action = 1
        #save original state
        lastState = newState
        newState[0], reward, done, info = environment.step(action)
        environment.render()
        # calculate true future reward
        Q1 = sess.run(output, feed_dict={x: newState})
        maxQ1 = np.max(Q1)
        targetQ = netOutput
        targetQ[action, 0] = reward + gamma * maxQ1
        # Train our network using target and predicted Q values
        sess.run(loss, feed_dict={x: lastState, y: targetQ.T})
        print(loss)
        if done:
            epsilon = 1. / ((iteration / 50) + 10)
            print("epsilon = " + str(epsilon))
            print(info)
            newState[0] = environment.reset()
