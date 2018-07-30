import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


class LogisticRegresionModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

    def create_placeholders(self, n_x, n_y):
        # START CODE HERE ### (approx. 2 lines)
        X = tf.placeholder(tf.float32, [n_x, None], name="X")
        Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
        ### END CODE HERE ###

        return X, Y

    def initialize_parameters(self, width, output, layer_count=2, net_width=30):

        # so that your "random" numbers match ours
        tf.set_random_seed(1)

        # START CODE HERE ### (approx. 6 lines of code)
        W = [None] * layer_count
        B = [None] * layer_count

        W[0] = tf.get_variable(
            "W1", [net_width, width], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        B[0] = tf.get_variable("b1", [net_width, 1],
                               initializer=tf.zeros_initializer())
        for i in range(1, layer_count-1):
            W[i] = tf.get_variable("W"+str(i+1), [net_width, net_width],
                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
            B[i] = tf.get_variable(
                "b"+str(i+1), [net_width, 1], initializer=tf.zeros_initializer())
        W[layer_count-1] = tf.get_variable("W"+str(layer_count), [
                                           output, net_width], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        B[layer_count-1] = tf.get_variable("b"+str(layer_count),
                                           [output, 1], initializer=tf.zeros_initializer())

        ### END CODE HERE ###
        parameters = {"W": W, "b": B}

        return parameters

    # GRADED FUNCTION: forward_propagation

    def forward_propagation(self, X, parameters):
        # Retrieve the parameters from the dictionary "parameters"
        W = parameters['W']
        b = parameters['b']
        # START CODE HERE ### (approx. 5 lines)
        Z = tf.add(tf.matmul(W[0], X), b[0])
        A = tf.nn.relu(Z)                                    # A1 = relu(Z1)
        n = len(W)
        for i in range(1, n-1):
            Z = tf.add(tf.matmul(W[i], A), b[i])
            A = tf.nn.relu(Z)
        Z = tf.add(tf.matmul(W[n-1], A), b[n-1])
        ### END CODE HERE ###

        return Z

    def compute_cost(self, Z3, Y):
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)

        # START CODE HERE ### (1 line of code)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))
        ### END CODE HERE ###

        return cost

    def model(self, learning_rate=0.0001,
              num_epochs=60, minibatch_size=32, print_cost=True):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3                                          # to keep consistent results
        # (n_x: input size, m : number of examples in the train set)
        (m, n_x) = self.X_train.shape
        # n_y : output size
        n_y = self.y_train.shape[0]
        costs = []
        test_acc = []
        train_acc = []
        X, Y = self.create_placeholders(n_x, 1)
        parameters = self.initialize_parameters(n_x, 1)
        Z3 = self.forward_propagation(X, parameters)
        cost = self.compute_cost(Z3, Y)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=config) as sess:
            # Run the initialization
            sess.run(init)
            # prediction accuracy
            predicted = tf.nn.sigmoid(Z3)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(predicted), Y), tf.float32))

            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                # number of minibatches of size minibatch_size in the train set
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = self.random_mini_batches(np.transpose(
                    self.X_train), np.matrix([self.y_train]), minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    # START CODE HERE ### (1 line)
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={
                                                 X: minibatch_X, Y: minibatch_Y})
                    ### END CODE HERE ###

                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(epoch_cost)
                    train_acc.append(accuracy.eval(
                        {X: np.transpose(self.X_train), Y: np.matrix([self.y_train])}))
                    test_acc.append(accuracy.eval(
                        {X: np.transpose(self.X_test), Y: np.matrix([self.y_test])}))

            # plot the cost
            plt.gca().set_color_cycle(['blue', 'green', 'red'])
            plt.xticks(np.arange(0, len(costs)+1, step=1.0))
            plt.plot(np.squeeze(costs))
            plt.plot(np.squeeze(train_acc))
            plt.plot(np.squeeze(test_acc))
            plt.legend(['Learning rate', 'train accuracy',
                        'test accuracy'], loc='upper left')
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print("Parameters have been trained!")

            print("Train Accuracy:", accuracy.eval(
                {X: np.transpose(self.X_train), Y: np.matrix([self.y_train])}))
            print("Test Accuracy:", accuracy.eval(
                {X: np.transpose(self.X_test), Y: np.matrix([self.y_test])}))

            return parameters

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]  # .reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k *
                                      mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:,
                                      num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:,
                                      num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
