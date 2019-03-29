import numpy as np
from layer import Affine, Sigmoid, Softmax_with_loss, ReLU, Identity_with_loss
from except_def import UnableToUseException

class neuralNetwork:
    def __init__(self):
        self.num_units = None
        self.layer_types = None
        self.hyperparams = None
        self.weight_init_std = 0.01
        #####
        self.layers = None
        self.x_train = None
        self.t_train = None
        #####
        self.x_estimate = None
        self.t_estimate = None
        #####
        self.loss = None
        self.accuracy = None
        #####
        self.optimizer = None
        self.diff_algorithm = None
        self.input_dim = None
        self.output_dim = None

    def set_structure(self,num_units, layer_types):
        self.num_units = num_units
        self.layer_types = layer_types

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def init_layers(self):
        methods_for_type = {'Affine': self.set_Affine, 'ReLU': self.set_ReLU, 'Sigmoid': self.set_Sigmoid, 'Softmax_with_loss': self.set_Softmax_with_loss, 'Identity_with_loss': self.set_Identity_with_loss}

        self.layers = []
        num_layer = len(self.layer_types)
        for i in range(num_layer):
            type = self.layer_types[i]
            layer = methods_for_type[type](i)
            self.layers.append(layer)

    def set_Affine(self, i):
        if i < len(self.num_units) - 1:
            n_row = self.num_units[i]
            n_col = self.num_units[i+1]
            W = self.weight_init_std * np.random.randn(n_row, n_col)
            b = np.zeros(n_col)
            return Affine(W, b)
        else:
            raise UnableToUseException('Affine layer cannot be the last layer')

    def set_Sigmoid(self, i):
        if i < len(self.num_units) - 1:
            return Sigmoid(self.num_units[i])
        else:
            raise UnableToUseException('Sigmoid layer cannot be the last layer')

    def set_Softmax_with_loss(self, i):
        if i == len(self.num_units) - 1:
            return Softmax_with_loss(self.num_units[i])
        else:
            raise UnableToUseException('Softmax_with_loss layer should be the last layer')

    def set_ReLU(self, i):
        if i < len(self.num_units) - 1:
            return ReLU(self.num_units[i])
        else:
            raise UnableToUseException('ReLU layer cannot be the last layer')

    def set_Identity_with_loss(self, i):
        if i == len(self.num_units) - 1:
            return Identity_with_loss(self.num_units[i])
        else:
            raise UnableToUseException('Identity_with_loss layer should be the last layer')

    def set_data(self, data_train, data_estimate):
        if len(data_train) == 2 and len(data_estimate) == 2:
            self.x_train = data_train[0]
            self.t_train = data_train[1]
            self.x_estimate = data_estimate[0]
            self.t_estimate = data_estimate[1]
        else:
            raise UnableToUseException('The data are not allowed: set_data')

    def predict(self, input):
        size = len(self.layers)
        x = input
        for i in range(size):
            x = self.layers[i].forward(x)
        p = np.argmax(x)
        return p

    def estimate(self):
        pass

    def learn(self):
        pass

    def back_prop(self, i):
        pass

    def update_network(self, i):
        pass

    def update(self, i):
        pass

    def show_loss(self):
        pass

    def show_accuracy(self):
        pass

#######

    def set_optimizer(self, optimizer):
        pass

    def set_diff_algorithm(self, diff_algorithm):
        pass

    def loss(x,t):
        pass

    def get_grad(self):
        pass

    def numerical_grad(self, x, t):
        pass

    def fout(self):
        pass

    def fin(self):
        pass
