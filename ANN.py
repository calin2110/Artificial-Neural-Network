import pickle

import matplotlib.pyplot as plt
import numpy as np

from Constants import BATCH_SIZE
from Layer import Layer
from Utils import Dataset
from Functions import Function


class ANN:
    __dataset: Dataset
    __layers: list[Layer]
    __average_loss: float

    def __init__(self):
        self.__dataset = Dataset()

    def initialise_neural_network(
            self,
            neurons_count: list[int],
            activation_functions: list[Function]
    ):
        if len(neurons_count) != len(activation_functions) + 1:
            raise Exception("plm")
        self.__layers = []
        for i in range(len(neurons_count[:-1])):
            layer: Layer = Layer(input_layer_neurons=neurons_count[i], output_layer_neurons=neurons_count[i + 1],
                                 activation_function=activation_functions[i])
            self.__layers.append(layer)

    def save_current_state(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __forward(self, values: np.array) -> np.array:
        for layer in self.__layers:
            values = layer.forward(values)
        return values

    def __backward(self, multiplier: np.array):
        for layer in reversed(self.__layers):
            multiplier = layer.backward(multiplier_matrix=multiplier)

    def __complete_pass(self, values: np.array, supposed_activations: np.array) -> float:
        results = self.__forward(values=values)
        loss = np.sum((supposed_activations - results) ** 2)
        self.__backward(multiplier=2 * (results - supposed_activations))
        return loss

    def __propagate_gradient_descent(self):
        for layer in self.__layers:
            layer.gradient_descent(self.__average_loss)

    def predict_result(self, values):
        predicted = self.__forward(values)
        return np.argmax(predicted)

    def train(self):
        iteration_count: int = 1

        # TODO: integrate ewa and ews
        xs = []
        ys = []
        while iteration_count < 5000:
            batch: list[tuple[int, np.array]] = self.__dataset.get_batch(BATCH_SIZE)
            total_loss: float = 0

            for image in batch:
                supposed_activations: np.array = np.zeros((10, 1))
                digit: int = image[0]
                supposed_activations[digit, 0] = 1

                values: np.array = image[1]
                total_loss += self.__complete_pass(values=values, supposed_activations=supposed_activations)

            self.__average_loss = total_loss / BATCH_SIZE
            print(f"Average loss is: {round(self.__average_loss, 3)} for iteration number {iteration_count}")
            xs.append(iteration_count)
            ys.append(self.__average_loss)
            iteration_count += 1
            self.__propagate_gradient_descent()

        plt.ylim(bottom=0, top=2)
        plt.plot(xs, ys)
        plt.show()

    def test(self):
        correct_answers = 0
        total_answers = len(self.__dataset.test_X)
        for i in range(total_answers):
            pixels_to_predict = self.__dataset.test_X[i]
            predicted_answer = np.argmax(self.__forward(pixels_to_predict))
            correct_answer = self.__dataset.test_y[i]
            if predicted_answer == correct_answer:
                correct_answers += 1
        accuracy = correct_answers / total_answers
        print(f"Accuracy: {accuracy}")

    def save_to_files(self, files_list: list[tuple[str, str, str]]):
        for i, tup in enumerate(files_list):
            self.__layers[i].save_to_files(tup[0], tup[1], tup[2])

    def read_from_files(self, files_list: list[tuple[str, str, str]]):
        self.__layers = [Layer(-1, -1) for _ in files_list]
        for i, tup in enumerate(files_list):
            self.__layers[i].read_from_files(tup[0], tup[1], tup[2])