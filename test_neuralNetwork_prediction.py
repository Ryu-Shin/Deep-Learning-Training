import unittest
from neuralNetwork import neuralNetwork
from layer import Affine, Sigmoid, Softmax_with_loss, ReLU, Identity_with_loss
import numpy as np
from dataset.mnist import load_mnist
import pickle
from unittest.mock import MagicMock
from except_def import UnableToUseException


class Test_NN_prediction(unittest.TestCase):

        def setUp(self):

            (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=True, one_hot_label=True)
            self.xtrain = x_train
            self.ttrain = t_train
            self.xtest = x_test
            self.ttest = t_test

            with open("sample_weight.pkl", 'rb') as f:
                self.network = pickle.load(f)
            self.W1 = self.network['W1']
            self.W2 = self.network['W2']
            self.W3 = self.network['W3']
            self.b1 = self.network['b1']
            self.b2 = self.network['b2']
            self.b3 = self.network['b3']
            self.b1 = self.b1.reshape(1, -1)
            self.b2 = self.b2.reshape(1, -1)
            self.b3 = self.b3.reshape(1, -1)

            self.layer1 = Affine(self.W1, self.b1)
            self.sigmoid1 = Sigmoid(self.b1.shape[1])
            self.layer2 = Affine(self.W2, self.b2)
            self.sigmoid2 = Sigmoid(self.b2.shape[1])
            self.layer3 = Affine(self.W3, self.b3)
            self.softmax_with_loss = Softmax_with_loss(self.b3.shape[1])

            self.target = neuralNetwork()
            self.target.layers = (self.layer1, self.sigmoid1, self.layer2, self.sigmoid2, self.layer3, self.softmax_with_loss)
            self.target.hyperparams = {'learning_rate': 0.1, 'iteration': 10000, 'size_batch': 100}
            self.target.x_train = self.xtrain
            self.target.t_train = self.ttrain
            self.target.x_estimate = self.xtest
            self.target.t_estimate = self.ttest

        def tearDown(self):
            pass


        def test_predict(self):
            label = (7, 2, 1, 0, 4, 1, 4, 9, 6, 9)
            for i in range(10):
                expect = label[i]
                actual = self.target.predict(self.target.x_estimate[i])
                self.assertEqual(actual, expect)
