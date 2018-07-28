import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


class SVMModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.W = tf.Variable(tf.random_normal(
            shape=[self.X_train.shape[1],1]))
        self.b = tf.Variable(tf.random_normal(
            shape=[1,1]))

    def create_placeholders(self):
        # START CODE HERE ### (approx. 2 lines)
        X = tf.placeholder(tf.float32, [None, self.X_train.shape[1]], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        ### END CODE HERE ###

        return X, Y

    def model(self, run_count=500, alpha=0.001, learning_rate=0.1, batch_size=64):
        
        X, Y = self.create_placeholders()

        #cost
        model_output = tf.subtract(tf.matmul(X, self.W), self.b)
        l2_norm = tf.reduce_sum(tf.square(self.W))
        alpha = tf.constant([alpha])
        classification_term = tf.reduce_mean(tf.maximum(
            0., tf.subtract(1., tf.multiply(model_output, Y))))
        cost = tf.add(classification_term, tf.multiply(alpha, l2_norm))

        #accuracy
        model_output = tf.subtract(tf.matmul(X, self.W), self.b)
        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

        # Initialize variables
        init = tf.global_variables_initializer()

        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(cost)
        
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
        with tf.Session(config=config) as sess:
            sess.run(init)
            # Training loop
            loss_vec = []
            train_accuracy = []
            test_accuracy = []

            for i in range(run_count):
                rand_index = np.random.choice(
                    len(self.X_train), size=batch_size)
                rand_x = self.X_train[rand_index]
                rand_y = np.transpose([self.y_train[rand_index]])
                sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

                temp_loss = sess.run(cost, feed_dict={
                                     X: rand_x, Y: rand_y})
                loss_vec.append(temp_loss)

                train_acc_temp = sess.run(accuracy, feed_dict={
                    X: self.X_train,
                    Y: np.transpose([self.y_train])})
                train_accuracy.append(train_acc_temp)

                test_acc_temp = sess.run(accuracy, feed_dict={
                    X: self.X_test,
                    Y: np.transpose([self.y_test])})
                test_accuracy.append(test_acc_temp)

                if (i + 1) % 100 == 0:
                    print('Step #{} test_acc = {}, train_acc = {}'.format(
                        str(i+1),
                        str(test_acc_temp),
                        str(train_acc_temp)
                    ))
                    print('Loss = ' + str(temp_loss))
        # Extract coefficients
        # W_final = sess.run(self.W)
        # b_final = sess.run(self.b)

        # Plot train/test accuracies
        plt.plot(train_accuracy, 'k-', label='Training Accuracy')
        plt.plot(test_accuracy, 'r--', label='Test Accuracy')
        plt.title('Train and Test Set Accuracies')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()
