import pickle

import numpy as np

from Constants import LEARNING_RATE, BATCH_SIZE
from Utils import generate_random_matrix
from Functions import Function


class Layer:
    # the weights
    __w: np.array

    # the biases
    __b: np.array

    # the activation function
    __f: Function

    __values: np.array

    # the saved value of the pass after a forward pass
    __pass: np.array

    # the activated pass
    __activation: np.array

    __total_delta_b: np.array

    __total_delta_w: np.array

    def __init__(self, input_layer_neurons: int, output_layer_neurons: int, activation_function: Function = None):
        self.__w = generate_random_matrix(lines=output_layer_neurons, columns=input_layer_neurons)
        self.__b = generate_random_matrix(lines=output_layer_neurons, columns=1)
        self.__f = activation_function
        self.__total_delta_b = np.zeros_like(self.__b)
        self.__total_delta_w = np.zeros_like(self.__w)

    def forward(self, values: np.array) -> np.array:
        self.__values = values
        self.__pass = np.dot(self.__w, values) + self.__b
        self.__activation = self.__f.forward(self.__pass)
        return self.__activation

    def backward(self, multiplier_matrix: np.array):
        delta_b = self.__f.backward(self.__pass) * multiplier_matrix
        self.__total_delta_b += delta_b
        self.__total_delta_w += np.dot(delta_b, self.__values.T)
        return np.dot(self.__w.T, delta_b)

    def gradient_descent(self, average_error: float):
        self.__w -= LEARNING_RATE * average_error * self.__total_delta_w / BATCH_SIZE
        self.__b -= LEARNING_RATE * average_error * self.__total_delta_b / BATCH_SIZE
        self.__total_delta_b = np.zeros_like(self.__b)
        self.__total_delta_w = np.zeros_like(self.__w)

    def read_from_files(self, weight_file: str, bias_file: str, function_file: str):
        with open(bias_file, "rb") as f:
            self.__b = pickle.load(f)
        with open(weight_file, "rb") as f:
            self.__w = pickle.load(f)
        with open(function_file, "rb") as f:
            self.__f = pickle.load(f)
        self.__total_delta_b = np.zeros_like(self.__b)
        self.__total_delta_w = np.zeros_like(self.__w)

    def save_to_files(self, weight_file: str, bias_file: str, function_file: str):
        with open(bias_file, "wb") as f:
            pickle.dump(self.__b, f)
        with open(weight_file, "wb") as f:
            pickle.dump(self.__w, f)
        with open(function_file, "wb") as f:
            pickle.dump(self.__f, f)
