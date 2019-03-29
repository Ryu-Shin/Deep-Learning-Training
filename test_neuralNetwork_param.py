import unittest
from neuralNetwork import neuralNetwork
from layer import Affine, Sigmoid, Softmax_with_loss, ReLU, Identity_with_loss
import numpy as np
from dataset.mnist import load_mnist
import pickle
from unittest.mock import MagicMock
from except_def import UnableToUseException

class Test_NN_param(unittest.TestCase):

        def setUp(self):
            (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=True, one_hot_label=False)
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

        def tearDown(self):
            pass

        def test_set_structure(self):
            val_num_units = (784, 50, 50, 100, 100, 10)
            val_layer_types = ('Affine', 'Sigmoid', 'Affine', 'Sigmoid', 'Affine', 'Softmax_with_loss')

            expect_num_units = (784, 50, 50, 100, 100, 10)
            expect_layer_types = ('Affine', 'Sigmoid', 'Affine', 'Sigmoid', 'Affine', 'Softmax_with_loss')

            network = neuralNetwork()
            network.set_structure(val_num_units, val_layer_types)
            actual_num_units = network.num_units
            actual_layer_types = network.layer_types

            self.assertEqual(actual_num_units, expect_num_units)
            self.assertEqual(actual_layer_types, expect_layer_types)

#### test for the case that len(num_units) != len(layer_types)
#### test for the case that the arguments are not tuples
#### test for the case that the last layer is not Softmax_with_loss or Identity_with_loss


        def test_set_hyperparams(self):
            val_hyperparams = {'learning_rate': 0.1, 'iteration': 10000, 'size_batch': 100}
            expect_hyperparams = {'learning_rate': 0.1, 'iteration': 10000, 'size_batch': 100}

            network = neuralNetwork()
            network.set_hyperparams(val_hyperparams)

            actual_hyperparams = network.hyperparams

            self.assertEqual(actual_hyperparams['learning_rate'], expect_hyperparams['learning_rate'])
            self.assertEqual(actual_hyperparams['iteration'], expect_hyperparams['iteration'])
            self.assertEqual(actual_hyperparams['size_batch'], expect_hyperparams['size_batch'])

#### test for the case that the argument is not dictionary


        def test_init_layers(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 784, 50, 784, 10)
            network.layer_types = ('Affine', 'Sigmoid', 'Affine', 'Sigmoid', 'Affine', 'Softmax_with_loss')
            network.hyperparams = {'learning_rate': 0.1, 'iteration': 10000, 'size_batch': 100}

            network.set_Affine = MagicMock(return_value = self.target.layers[0])
            network.set_Sigmoid = MagicMock(return_value = self.target.layers[1])
            network.set_Softmax_with_loss = MagicMock(return_value = self.target.layers[5])

            network.init_layers()

            actual_layers = network.layers
            expect_layers = self.target.layers
            self.assertEqual(len(actual_layers), len(expect_layers))
            for i in range(len(expect_layers)):
                self.assertEqual(type(actual_layers[i]), type(expect_layers[i]))

            place_expected = (0, 1, 0, 1, 0, 5)
            for i in range(len(actual_layers)):
                num = place_expected[i]
                self.assertEqual(actual_layers[i].num_unit, expect_layers[num].num_unit)

        def test_set_Affine(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)

            expect_0_shape = self.target.layers[0].W.shape
            expect_2_shape = self.target.layers[2].W.shape
            expect_4_shape = self.target.layers[4].W.shape

            expect_0_b = np.zeros(50)
            expect_0_b = expect_0_b.reshape(1, -1)
            expect_2_b = np.zeros(100)
            expect_2_b = expect_2_b.reshape(1, -1)
            expect_4_b = np.zeros(10)
            expect_4_b = expect_4_b.reshape(1, -1)

            actual_0 = network.set_Affine(0)
            actual_2 = network.set_Affine(2)
            actual_4 = network.set_Affine(4)

            actual_0_shape = actual_0.W.shape
            actual_2_shape = actual_2.W.shape
            actual_4_shape = actual_4.W.shape
            actual_0_b = actual_0.b
            actual_2_b = actual_2.b
            actual_4_b = actual_4.b

            self.assertEqual(actual_0_shape, expect_0_shape)
            self.assertEqual(actual_2_shape, expect_2_shape)
            self.assertEqual(actual_4_shape, expect_4_shape)
            np.testing.assert_array_equal(actual_0_b, expect_0_b)
            np.testing.assert_array_equal(actual_2_b, expect_2_b)
            np.testing.assert_array_equal(actual_4_b, expect_4_b)

        def test_set_Affine_end_of_array(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            with self.assertRaises(UnableToUseException) as er:
                self.smpl = network.set_Affine(5)
            self.assertEqual(er.exception.args[0], 'Affine layer cannot be the last layer')

        def test_set_Sigmoid(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            expect_1 = self.target.layers[1]
            expect_3 = self.target.layers[3]
            actual_1 = network.set_Sigmoid(1)
            actual_3 = network.set_Sigmoid(3)

            self.assertEqual(type(actual_1), type(expect_1))
            self.assertEqual(type(actual_3), type(expect_3))
            self.assertEqual(actual_1.num_unit, expect_1.num_unit)
            self.assertEqual(actual_3.num_unit, expect_3.num_unit)

        def test_set_Sigmoid_end_of_array(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            with self.assertRaises(UnableToUseException) as er:
                self.smpl = network.set_Sigmoid(5)
            self.assertEqual(er.exception.args[0], 'Sigmoid layer cannot be the last layer')

        def test_set_Softmax_with_loss(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            expect_5 = self.target.layers[5]
            actual_5 = network.set_Softmax_with_loss(5)

            self.assertEqual(type(actual_5), type(expect_5))
            self.assertEqual(actual_5.num_unit, expect_5.num_unit)

        def test_set_Softmax_with_loss_not_end_of_array(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            with self.assertRaises(UnableToUseException) as er:
                self.smpl = network.set_Softmax_with_loss(4)
            self.assertEqual(er.exception.args[0], 'Softmax_with_loss layer should be the last layer')

        def test_set_ReLU(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            expect_1 = ReLU(50)
            actual_1 = network.set_ReLU(1)

            self.assertEqual(type(actual_1), type(expect_1))
            self.assertEqual(actual_1.num_unit, expect_1.num_unit)

        def test_set_ReLU_end_of_array(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            with self.assertRaises(UnableToUseException) as er:
                self.smpl = network.set_ReLU(5)
            self.assertEqual(er.exception.args[0], 'ReLU layer cannot be the last layer')

        def test_set_Identity_with_loss(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            expect_5 = Identity_with_loss(10)
            actual_5 = network.set_Identity_with_loss(5)

            self.assertEqual(type(actual_5), type(expect_5))
            self.assertEqual(actual_5.num_unit, expect_5.num_unit)

        def test_set_Identity_with_loss_not_end_of_array(self):
            network = neuralNetwork()
            network.num_units = (784, 50, 50, 100, 100, 10)
            with self.assertRaises(UnableToUseException) as er:
                self.smpl = network.set_Identity_with_loss(4)
            self.assertEqual(er.exception.args[0], 'Identity_with_loss layer should be the last layer')

        def test_set_data(self):
            network = neuralNetwork()
            data_train = (self.xtrain, self.ttrain)
            data_estimate = (self.xtest, self.ttest)
            network.set_data(data_train, data_estimate)

            expect_x_train = data_train[0]
            expect_t_train = data_train[1]
            expect_x_estimate = data_estimate[0]
            expect_t_estimate = data_estimate[1]

            actual_x_train = network.x_train
            actual_t_train = network.t_train
            actual_x_estimate = network.x_estimate
            actual_t_estimate = network.t_estimate

            self.assertEqual(type(actual_x_train), type(expect_x_train))
            self.assertEqual(type(actual_t_train), type(expect_t_train))
            self.assertEqual(type(actual_x_estimate), type(expect_x_estimate))
            self.assertEqual(type(actual_x_estimate), type(expect_x_estimate))

        def test_set_data_wrong_size(self):
            network = neuralNetwork()
            data_train = (self.xtrain,)
            data_estimate = (self.xtest,)
            with self.assertRaises(UnableToUseException) as er:
                network.set_data(data_train, data_estimate)
            self.assertEqual(er.exception.args[0], 'The data are not allowed: set_data')

#### test for the case of regression






if __name__ == "__main__":
    unittest.main()
