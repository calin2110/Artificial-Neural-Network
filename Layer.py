import pickle

import numpy as np

import Functions
from Constants import LEARNING_RATE, BATCH_SIZE
from OptimizationType import GradientDescent, OptimizationType, OptimizationFactory
from Regularization import Regularization, RegularizationType, factory
from Utils import generate_random_matrix
from Functions import Function


class Layer:
    # gradient descent optimization used
    __w_gradient_descent: GradientDescent
    __b_gradient_descent: GradientDescent

    __regularization: Regularization

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

    def __init__(self,
                 input_layer_neurons: int,
                 output_layer_neurons: int,
                 activation_function: Function = Functions.Sigmoid,
                 gd_optimization: OptimizationType = OptimizationType.STOCHASTIC,
                 regularization_type: RegularizationType = RegularizationType.NONE
                 ):
        self.__f = activation_function
        self.__regularization = factory.create_regularization(regularization_type)
        self.__w = generate_random_matrix(lines=output_layer_neurons, columns=input_layer_neurons)
        self.__w_gradient_descent = \
            OptimizationFactory.create_optimization(optimization=gd_optimization, shape=self.__w.shape)
        self.__total_delta_w = np.zeros_like(self.__w)
        self.__b = generate_random_matrix(lines=output_layer_neurons, columns=1)
        self.__b_gradient_descent = \
            OptimizationFactory.create_optimization(optimization=gd_optimization, shape=self.__b.shape)
        self.__total_delta_b = np.zeros_like(self.__b)

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

    def gradient_descent(self):
        self.__w -= LEARNING_RATE * self.__regularization.backward(self.__w)
        self.__w -= LEARNING_RATE * self.__w_gradient_descent.make_step(self.__total_delta_w / BATCH_SIZE)
        self.__b -= LEARNING_RATE * self.__regularization.backward(self.__b)
        self.__b -= LEARNING_RATE * self.__b_gradient_descent.make_step(self.__total_delta_b / BATCH_SIZE)
        self.__total_delta_b = np.zeros_like(self.__b)
        self.__total_delta_w = np.zeros_like(self.__w)

    def get_regularization_value(self) -> float:
        return self.__regularization.forward(self.__w) + self.__regularization.forward(self.__b)

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
