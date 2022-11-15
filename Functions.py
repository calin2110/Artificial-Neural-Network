import numpy as np


# interface defining the functionalities an activation function should have
class Function:
    def forward(self, x_0):
        pass

    def backward(self, x_0):
        pass


class Sigmoid(Function):
    def forward(self, x_0):
        return 1 / (1 + np.exp(-x_0))

    def backward(self, x_0):
        return np.exp(-x_0) / ((1 + np.exp(-x_0)) ** 2)


class Linear(Function):
    __a: float
    __b: float

    def __init__(self, a: float = 1, b: float = 0):
        self.__a = a
        self.__b = b

    def forward(self, x_0):
        return self.__a * x_0 + self.__b

    def backward(self, x_0):
        return self.__a


class ReLU(Function):
    def forward(self, x_0):
        return np.maximum(0, x_0)

    def backward(self, x_0):
        gradient: np.array = np.zeros_like(x_0)
        gradient[x_0 > 0] = 1
        gradient[x_0 <= 0] = 0
        return gradient


class TanH(Function):
    def forward(self, x_0):
        return (1 - np.exp(-2*x_0)) / (1 + np.exp(-2*x_0))

    def backward(self, x_0):
        return 4 * np.exp(-2*x_0) / ((1 + np.exp(-2*x_0)) ** 2)


class LeakyReLU(Function):
    def forward(self, x_0):
        return np.maximum(0, x_0)

    def backward(self, x_0):
        gradient: np.array = np.zeros_like(shape=x_0.shape)
        gradient[x_0 > 0] = 1
        gradient[x_0 <= 0] = 0.1
        return gradient


class Functions:
    sigmoid = Sigmoid()
    linear = Linear()
    relu = ReLU()
    tanh = TanH()
