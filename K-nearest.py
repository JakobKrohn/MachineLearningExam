import numpy as np
import PIL.Image as pil

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("model_data", one_hot=True)

#Sends 100 pictures to the algorithm.
batch = 100
#Controls the size of the parameters and rates.
learning_rate= 0.01
#Goes throug 10 cycles of the algoritmh
training_epochs= 10


#How big the images are. 24x24
x = tf.placeholder(tf.float32, shape=[None, 784])
#which number it can be.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Wheight
W = tf.Variable(tf.zeros([784, 10]))
#Bias
b = tf.Variable(tf.zeros([10]))

#Initializing the model
y = tf.nn.softmax(tf.matmul(x, W) + b)


# a cost function is a difference between the predicted value
# and the actual value that we are trying to minimize to improve the accuracy of the model.

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Implementing Gradient Descent  Algorithm

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


for epoch in range(training_epochs):
    batch_count = int(mnist.train.num_examples/batch)
    for i in range(batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch)


sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})

if epoch % 2 == 0:
    print("Epoch: ", epoch)
    print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("Model Execution complete")




