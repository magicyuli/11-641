import numpy as np
import time

__author__ = 'lee'


class RMLRClassifier:
    def __init__(self, x=None, y=None, w=None):
        """
        Init the classifier, either by x and y, or by the trained w
        :param x: n by m
        :param y: n by 5
        :param w: 5 by m
        :return: an object
        """
        self._x = x
        self._y = y
        if w is None:
            w = np.matrix(np.ones((5, x.shape[1])) * (1.0 / x.shape[1]))
        self._w = w

    def _prob(self, batch_x):
        """
        Given some samples, compute the soft max probability
        of they belong to each class
        :param batch_x: some data samples
        :return: a probability matrix
        """
        P = np.exp(batch_x * self._w.T)
        iv_sum = 1 / np.sum(P, 1)
        P = np.multiply(P, iv_sum)
        return P

    def train(self):
        # hyperparameters
        stop = 0.00001
        beta = 1.2
        lambd = 0.03
        sample_size = self._y.shape[0]
        batch_size = 900

        # helper variables
        batch_cnt = (sample_size + batch_size - 1) / batch_size
        converged = False
        n = 0
        epoch = 1
        last_log_l_h = -float("inf")
        while not converged:
            print "=============start epoch %d=============" % epoch
            s_t = time.time()
            tmp_l_h = 0
            for i in range(0, batch_cnt):
                alpha = (100 + n) ** -beta
                start = i * batch_size
                end = min((i + 1) * batch_size, sample_size)
                # mini-batch x: k by m
                batch_x = self._x[start:end, :]
                # mini-batch y: k by 5
                batch_y = self._y[start:end, :]
                # probability matrix A: k by 5
                A = self._prob(batch_x)
                A_log = np.log(A)
                tmp_l_h += np.sum(batch_y.multiply(A_log))
                # B: k by 5
                B = batch_y - A
                # w: 5 by m
                self._w += alpha * (B.T * batch_x - lambd * self._w)
                n += 1
            print str((tmp_l_h - last_log_l_h) / -last_log_l_h)
            # test the change in log likelihood
            if tmp_l_h - last_log_l_h > 0 \
                    and (tmp_l_h - last_log_l_h) / -last_log_l_h < stop:
                converged = True
                print "converged..."
            last_log_l_h = tmp_l_h
            print "Alpha: %f" % alpha
            print "Log likelihood: %f" % last_log_l_h
            print "Time used: %f" % (time.time() - s_t)
            print "=============end epoch %d=============" % epoch
            epoch += 1

    def get_w(self):
        """
        Return the trained parameters.
        :return: the trained parameters.
        """
        return self._w

    def predict(self, x):
        """
        Predict the classes of samples.
        :param x: data samples
        :return: the hard and soft classifications
        """
        prob = self._prob(x)
        hard_clz = np.argmax(prob, 1)

        # soft predictions
        # [0,1,2,3,4]'
        coef = np.matrix(np.arange(0, 5)).T
        soft_clz = prob * coef

        return hard_clz, soft_clz

