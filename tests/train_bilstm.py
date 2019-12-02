# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# +
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

#Placeholders
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

#Variables
W = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))

weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                      mean = 0,
                                      stddev=0.01,
                                      name='weights'))

bias = tf.Variable(tf.random_normal([1, numLabels],
                                      mean = 0,
                                      stddev=0.01,
                                      name='bias'))

#Operations
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

#Defining the learning rate
numEpochs = 700
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                         global_step=1,
                                         decay_steps=trainX.shape[0],
                                         decay_rate=0.95,
                                         staircase=True)
#Cost function 
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

#Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# +
# Create a tensorflow session
sess = tf.Session()

#Init weights
init_OP = tf.global_variables_initializer()

# Init all tensorflow variables
sess.run(init_OP)

# +
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, 'float'))
activation_summary_OP = tf.summary.scalar("output", activation_OP)
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

#merge summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])
basic_summ = tf.summary.merge([cost_summary_OP])
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# +
#Reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

#Training Epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        #Occasional reports
        epoch_values.append(i)
        summ, train_accuracy, newCost = sess.run([basic_summ, accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
        accuracy_values.append(train_accuracy)
        cost_values.append(newCost)
        writer.add_summary(summ)
        diff = abs(newCost - cost)
        cost = newCost
        
        print('step: %d, training accuracy %g, cost %g, change in cost %g'%(i, train_accuracy, newCost, diff))
        
print("final accuracy on test set: %s", str(sess.run(accuracy_OP, 
                                                    feed_dict={X: trainX, yGold: trainY})))
    
# -

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()


