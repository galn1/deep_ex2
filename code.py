import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def network_model(data, perform_dropout=False):
    """
    Runs the network
    Parameters:
        data (tf.placeholder): the input data
        perform_dropout (boolean): when True the dropout is performed.
    Output:
        the result of the network
    """

    # Input Layer.
    input_layer = tf.reshape(data, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same")

    # TODO: add norm

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same")

    # TODO: add norm
    tf.reduce_max()

    # Pooling Layer #1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer #1
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024)

    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=perform_dropout)

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dropout, units=1024)

    # the dense for the softmax
    predictions = tf.layers.dense(inputs=dense2, units=10)

    return predictions


def train_network():
    """
    trains the network 
    """

    # runs the network
    prediction = network_model(x)
    # TODO: use tf.logging
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            if (epoch + 1) % 250 == 0:
                print('Epoch', epoch + 1, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# the MNIST data set
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# batch size
batch_size = 100

# number of iterations
n_epochs = 5000

# the learning rate
learning_rate = 0.001

train_network()
