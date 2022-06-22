import random

import numpy as np
from PIL import Image
from keras.datasets import mnist

from Constants import MAX_PIXEL_VALUE, MAX_WEIGHT_VALUE, MIN_WEIGHT_VALUE


class Dataset:
    __train_X: np.array
    __train_y: np.array

    test_X: np.array
    test_y: np.array

    __current_position: int

    def __init__(self):
        (train_X, self.__train_y), (test_X, self.test_y) = mnist.load_data()

        self.__train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2], 1))
        self.__train_X = self.__train_X / MAX_PIXEL_VALUE

        self.test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] * test_X.shape[2], 1))
        self.test_X = self.test_X / MAX_PIXEL_VALUE
        self.__current_position = 0


    def get_batch(self, batch_size: int) -> list[tuple[int, np.array]]:
        train_batch = self.__train_X[self.__current_position: self.__current_position + batch_size]
        labels_batch = self.__train_y[self.__current_position: self.__current_position + batch_size]
        self.__current_position = (self.__current_position + batch_size) % self.__train_X.shape[0]
        return zip(labels_batch, train_batch)


def generate_random_matrix(lines: int, columns: int, min_value: float = MIN_WEIGHT_VALUE, max_value: float = MAX_WEIGHT_VALUE) -> np.array:
    # throw exception if max_value > min_value
    # return np.random.normal(0, 1, size=(lines, columns))
    # return np.zeros((lines, columns))
    return np.random.uniform(min_value, max_value, size=(lines, columns))


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
