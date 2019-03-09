import numpy as np

permitted_int = [int, np.int8, np.int16, np.int32, np.int64]
permitted_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
permitted_float = [float, np.float16, np.float32, np.float64]
permitted_complex = [np.complex64, np.complex128]

permitted = []
permitted.extend(permitted_int)
permitted.extend(permitted_uint)
permitted.extend(permitted_float)
permitted.extend(permitted_complex)

def identity_function(x):
    type_x = type(x)
    if type_x in permitted:
        return x
    elif  type_x == np.ndarray and x.dtype in permitted:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x
    else:
        raise ValueError('The value is not permitted : identity_function')


def softmax(x):
    type_x = type(x)
    if  type_x == np.ndarray and x.dtype in permitted:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = np.zeros(x.shape)
        for i in range(x.shape[0]):
            xi = x[i]
            constant = np.max(xi)
            exp_val = np.exp(xi-constant)
            sum_exp = np.sum(exp_val)
            yi = exp_val / sum_exp
            y[i] = yi
        return y
    elif type_x == np.ndarray:
        raise ValueError('The value is not permitted : softmax')
    else:
        raise ValueError('Not a ndarray')

def numerical_diff(f,x):
    if  type(x) in permitted:
        h = 1e-4
        h1 = x-h
        h2 = x+h
        delta = 2*h
        diff = (f(h2) - f(h1)) / delta
        return diff
    else:
        raise ValueError('The value is not permitted : numerical_diff')
