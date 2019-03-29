import numpy as np
from except_def import UnableToUseException
import utility as util

permitted_Layer = []
permitted_Layer.extend([int])

permitted_Affine = []
permitted_Affine.extend([np.int8, np.int16, np.int32, np.int64])
permitted_Affine.extend([np.uint8, np.uint16, np.uint32, np.uint64])
permitted_Affine.extend([np.float16, np.float32, np.float64])



class Layer:
    def __init__(self, num_unit):
        if type(num_unit) in permitted_Layer and num_unit >= 0:
            self.num_unit = num_unit
        else:
            raise ValueError('The value is not permitted : constructor of Layer')

    def forward(self):
        raise UnableToUseException('Uable to use this method: Layer')
    def backward(self):
        raise UnableToUseException('Uable to use this method: Layer')

    def check_input_forward(self, x):
        if type(x) == np.ndarray and x.dtype in permitted_Affine and ((x.ndim, x.size) == (1, self.num_unit) or (x.ndim, x[0].size) == (2, self.num_unit)):
            return True
        else:
            return False

    def check_input_backward(self, x):
        if type(x) == np.ndarray and x.dtype in permitted_Affine:
            return True
        else:
            return False


    def reshape_input(self, x):
        if type(x) == np.ndarray and x.ndim == 1:
            return x.reshape(1, -1)
        else:
            return x

class Affine(Layer):
    def __init__(self, W, b):
        super(Affine, self).__init__(0)
        self.W = None
        self.b = None
        self.x = None
        self.dW = None
        self.db = None
        b = self.reshape_input(b)

        if (type(W), type(b)) == (np.ndarray ,np.ndarray) and W.dtype in permitted_Affine and b.dtype in permitted_Affine and b.ndim == 2 and (b.shape[0], W.shape[1]) == (1, b.shape[1]):
            self.W = W
            self.b = b
            self.num_unit = W.shape[0]
        else:
            raise ValueError('The value is not permitted : constructor of Affine')

    def forward(self, x):
        if self.check_input_forward(x):
            x = self.reshape_input(x)
            self.x = x
            out = np.dot(x, self.W) + self.b
            return out
        else:
            raise ValueError('The value is not permitted : forward in Affine')

    def backward(self, dout):
        if self.check_input_backward(dout):
            dout = self.reshape_input(dout)
            dx = np.dot(dout, self.W.T)
            self.dx = dx
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0) # axis=0 で行方向(0方向)に加算する指定をしている
            self.db = self.reshape_input(self.db)
            return dx
        else:
            raise ValueError('The value is not permitted : backward in Affine')

    def update(self, optimizer):
        optimizer.update(self.W, self.b, self.dW, self.db)
        pass

class ReLU(Layer):
    def __init__(self, num_unit):
        super(ReLU, self).__init__(num_unit)
        self.mask = None

    def forward(self, x):
        if self.check_input_forward(x):
            x = self.reshape_input(x)
            self.mask = (x <= 0)
            x[self.mask] = 0
            return x
        else:
            raise ValueError('The value is not permitted : forward in ReLU')

    def backward(self, dout):
        if self.check_input_backward(dout):
            dout = self.reshape_input(dout)
            dx = dout
            dx[self.mask] = 0
            return dx
        else:
            raise ValueError('The value is not permitted : backward in ReLU')


class Sigmoid(Layer):
    def __init__(self, num_unit):
        super(Sigmoid, self).__init__(num_unit)
        self.y = None

    def forward(self, x):
        if self.check_input_forward(x):
            x = self.reshape_input(x)
            y = 1 / (1 + np.exp(-x))
            self.y = y
            return y
        else:
            raise ValueError('The value is not permitted : forward in Sigmoid')

    def backward(self, dout):
        if self.check_input_backward(dout):
            dout = self.reshape_input(dout)
            dx = dout * self.y * (1 - self.y)
            return dx
        else:
            raise ValueError('The value is not permitted : backward in Sigmoid')

class Softmax_with_loss(Layer):
    def __init__(self, num_unit):
        super(Softmax_with_loss, self).__init__(num_unit)
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x):
        if self.check_input_forward(x):
            x = self.reshape_input(x)
            y = util.softmax(x)
            return y
        else:
            raise ValueError('The value is not permitted : forward in Softmax_with_loss')

    def forward_with_loss(self, x, t):
        if self.check_input_forward(x) and self.check_input_forward(t):
            x = self.reshape_input(x)
            t = self.reshape_input(t)
            self.y = util.softmax(x)
            self.t = t
            temp = np.exp(x)
            delta = 1e-10
            batch_size = x.shape[0]
            self.loss = - np.sum(t * np.log(self.y + delta)) / batch_size # softmax
            return self.loss
        else:
            raise ValueError('The value is not permitted : forward in Softmax_with_loss')

    def backward(self):
        batch_size = self.y.shape[0]
        dx = (self.y-self.t) / batch_size
        return dx

class Identity_with_loss(Layer):
    def __init__(self, num_unit):
        super(Identity_with_loss, self).__init__(num_unit)
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x):
        pass

    def backward(self, dout):
        pass
