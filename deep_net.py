# '''
#
# // Froward Pass
# input > weight --> Input hidden l 1 (Activation Function) > weights --> hidden l 2 (Activation Function) > weight
# --> output Layer
#
# compare output to intended output > cost function (cross entropy)
# optimization function (optimizer) > minimizer cost (AdamOptimizer... SGridentDecent, AdaGrad 8diffrent type is there)
#
# ## After Getting the error ##
#
# We have to go back and say correct those values(adjust weight)
#
# ### Backpropagation
#
# feed forward + backpropagation ==== eppch (10 or 20 times)
#
#
# '''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9
# '''
# output will be like
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0,0]
# 4 = [0,0,0,0,1,0,0,0,0,0]
# 5 = [0,0,0,0,0,1,0,0,0,0]
# 6 = [0,0,0,0,0,0,1,0,0,0]
# 7 = [0,0,0,0,0,0,0,1,0,0]
# 8 = [0,0,0,0,0,0,0,0,1,0]
# 9 = [0,0,0,0,0,0,0,0,0,1]
# '''

n_nodes_hl1 = 500  # No of hidden node numbers for hidden layer 1
n_nodes_hl2 = 500  # No of hidden node numbers for hidden layer 2
n_nodes_hl3 = 500  # No of hidden node numbers for hidden layer 3

n_classes = 10  # [ 0...9 total classes 10,, it can be determined by mnist dataset too ]
batch_size = 100  # [it's gonna feed the network with 100 feature at a time. and another batch of images]

# height * weight
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    # Why Biases: If the input_data is 0 so after multiplying with weight it will be 0 again.

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # ( input_data * weights ) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)  # Rectifier Linear (Activation Function) same as sGradient Decent

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    # learning rate = 0.001 so we are not modifying
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles  feed forward + backward(backpropagation)
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Tranning -----------

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'losss:', epoch_loss)

        # Testing -----------

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
