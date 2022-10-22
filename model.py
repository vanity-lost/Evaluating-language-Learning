import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SVR(object):
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def fit(self, X, y, epochs=100, learning_rate=0.1):
        self.sess = tf.Session()

        feature_len = X.shape[-1] if len(X.shape) > 1 else 1

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.X = tf.placeholder(dtype=tf.float32, shape=(None, feature_len))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        self.W = tf.Variable(tf.random_normal(shape=(feature_len, 1)))
        self.b = tf.Variable(tf.random_normal(shape=(1,)))

        self.y_pred = tf.matmul(self.X, self.W) + self.b

        # Second part of following equation, loss is a function of how much the error exceeds a defined value, epsilon
        # Error lower than epsilon = no penalty.
        self.loss = tf.norm(self.W)/2 + tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))

        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            loss = self.sess.run(
                self.loss,
                {
                    self.X: X,
                    self.y: y
                }
            )
            print("{}/{}: loss: {}".format(i + 1, epochs, loss))

            self.sess.run(
                opt_op,
                {
                    self.X: X,
                    self.y: y
                }
            )

        return self

    def predict(self, X, y=None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        y_pred = self.sess.run(
            self.y_pred,
            {
                self.X: X
            }
        )
        return y_pred
