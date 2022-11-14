import numpy as np

from Constants import BATCH_SIZE, LEARNING_RATE, MODEL_FILE, ERROR_THRESHOLD, BIAS_FOR_WRONG_VALUE, \
    BIAS_FOR_CORRECT_VALUE, HIDDEN_LAYER_NEURONS
from Utils import Function, Sigmoid, generate_random_matrix, Linear, Dataset
import pickle
import matplotlib.pyplot as plt

class NeuralNetwork:
    # weights from the input layer to the first hidden layer
    # should be first_hidden_layers * input_layers
    __w1: np.array

    # biases of the first hidden layer
    # is a column with first_hidden_layers lines
    __b1: np.array

    # weights from the first hidden layer to the second hidden layer
    # should be second_hidden_layers * first_input_layers
    __w2: np.array

    # biases of the second hidden layer
    # is a column with second_hidden_layers lines
    __b2: np.array

    # weights from the second hidden layer to the output layer
    # should be x * y
    __wo: np.array

    # biases of the output layer
    # is a column with output_layers lines
    __bo: np.array

    # the activation function for each node
    # we will assume all the nodes are activated with the same function
    __f: Function

    __dataset: Dataset

    def __init__(self):
        pass

    def initialise_neural_network(self, input_layers: int = 28 * 28, first_hidden_layers: int = HIDDEN_LAYER_NEURONS,
                                  second_hidden_layers: int = HIDDEN_LAYER_NEURONS,
                                  output_layers: int = 10, activation_function: Function = Sigmoid()):
        self.__f = activation_function

        self.__w1 = generate_random_matrix(lines=first_hidden_layers, columns=input_layers)
        self.__b1 = generate_random_matrix(lines=first_hidden_layers, columns=1)

        self.__w2 = generate_random_matrix(lines=second_hidden_layers, columns=first_hidden_layers)
        self.__b2 = generate_random_matrix(lines=second_hidden_layers, columns=1)

        self.__wo = generate_random_matrix(lines=output_layers, columns=second_hidden_layers)
        self.__bo = generate_random_matrix(lines=output_layers, columns=1)

        self.__dataset = Dataset()

    def save_state(self, filename: str):
        with open("w1.nn", "wb") as f:
            pickle.dump(self.__w1, f)
        with open("b1.nn", "wb") as f:
            pickle.dump(self.__b1, f)
        with open("w2.nn", "wb") as f:
            pickle.dump(self.__w2, f)
        with open("b2.nn", "wb") as f:
            pickle.dump(self.__b2, f)
        with open("wo.nn", "wb") as f:
            pickle.dump(self.__wo, f)
        with open("bo.nn", "wb") as f:
            pickle.dump(self.__bo, f)

    # values has to be a column vector with input_layers lines
    def forward_pass(self, values: np.array, supposed_activations: np.array) -> dict[str, any]:
        activations = self.__find_activations_for_values(values=values)

        first_pass = activations["first_pass"]
        first_activation = activations["first_activation"]

        second_pass = activations["second_pass"]
        second_activation = activations["second_activation"]

        output_pass = activations["output_pass"]
        output_activation = activations["output_activation"]

        error = np.sum((supposed_activations - output_activation) ** 2)

        delta_bo = 2 * (output_activation - supposed_activations) * self.__f.backward(output_pass)
        delta_wo = np.dot(delta_bo, second_activation.T)

        delta_b2 = self.__f.backward(second_pass) * np.dot(self.__wo.T, delta_bo)
        delta_w2 = np.dot(delta_b2, first_activation.T)

        delta_b1 = self.__f.backward(first_pass) * np.dot(self.__w2.T, delta_b2)
        delta_w1 = np.dot(delta_b1, values.T)

        return {'error': error, 'delta_bo': delta_bo, 'delta_wo': delta_wo, 'delta_b2': delta_b2, 'delta_w2': delta_w2,
                'delta_b1': delta_b1, 'delta_w1': delta_w1}

    def __find_activations_for_values(self, values: np.array) -> dict[str, np.array]:
        first_pass = np.dot(self.__w1, values) + self.__b1
        first_activation = self.__f.forward(first_pass)

        second_pass = np.dot(self.__w2, first_activation) + self.__b2
        second_activation = self.__f.forward(second_pass)

        output_pass = np.dot(self.__wo, second_activation) + self.__bo
        output_activation = self.__f.forward(output_pass)
        return {"first_pass": first_pass, "first_activation": first_activation, "second_pass": second_pass,
                "second_activation": second_activation, "output_pass": output_pass, "output_activation": output_activation}

    def train(self):
        iteration_count: int = 1
        good_enough: bool = False

        xs = []
        ys = []
        while not good_enough:
            batch: list[tuple[int, np.array]] = self.__dataset.get_batch(BATCH_SIZE)
            total_error: float = 0
            total_delta_bo: np.array = np.zeros_like(self.__bo)
            total_delta_wo: np.array = np.zeros_like(self.__wo)
            total_delta_b2: np.array = np.zeros_like(self.__b2)
            total_delta_w2: np.array = np.zeros_like(self.__w2)
            total_delta_b1: np.array = np.zeros_like(self.__b1)
            total_delta_w1: np.array = np.zeros_like(self.__w1)

            for image in batch:
                supposed_activations: np.array = np.zeros((10, 1))
                digit: int = image[0]
                supposed_activations[digit, 0] = 1

                values: np.array = image[1]
                result = self.forward_pass(values=values, supposed_activations=supposed_activations)
                total_error += result["error"]
                total_delta_bo += result["delta_bo"]
                total_delta_wo += result["delta_wo"]
                total_delta_b2 += result["delta_b2"]
                total_delta_w2 += result["delta_w2"]
                total_delta_b1 += result["delta_b1"]
                total_delta_w1 += result["delta_w1"]

            average_error = total_error / BATCH_SIZE
            print(f"Average error is: {round(average_error, 3)} for iteration number {iteration_count}")
            xs.append(iteration_count)
            ys.append(average_error)
            iteration_count += 1
            self.__bo -= LEARNING_RATE * average_error * total_delta_bo / BATCH_SIZE
            self.__wo -= LEARNING_RATE * average_error * total_delta_wo / BATCH_SIZE

            self.__b2 -= LEARNING_RATE * average_error * total_delta_b2 / BATCH_SIZE
            self.__w2 -= LEARNING_RATE * average_error * total_delta_w2 / BATCH_SIZE

            self.__b1 -= LEARNING_RATE * average_error * total_delta_b1 / BATCH_SIZE
            self.__w1 -= LEARNING_RATE * average_error * total_delta_w1 / BATCH_SIZE
            good_enough = average_error < ERROR_THRESHOLD

            # if average_error < best_error / 2:
            #     self.save_state(MODEL_FILE)
            #     with open("thresholds.txt", "w") as f:
            #         f.write(str(average_error) + " " + str(iteration_count))
            #     best_error = average_error

        plt.plot(xs, ys)
        plt.show()
        self.save_state(MODEL_FILE)

    def predict_result(self, pixels: np.array) -> tuple[int, float]:
        result: np.array = self.__find_activations_for_values(pixels)["output_activation"]
        result = np.ndarray.flatten(result)
        max_percentage: float = 0.0
        chosen_digit: int = -1
        for i in range(10):
            if result[i] > max_percentage:
                max_percentage = result[i]
                chosen_digit = i
        return chosen_digit, max_percentage

    def test_on_test_data(self):
        correct_answers = 0
        total_answers = len(self.__dataset.test_X)
        answers = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        correctly_predicted = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for i in range(total_answers):
            pixels_to_predict = self.__dataset.test_X[i]
            predicted_answer = self.predict_result(pixels_to_predict)
            correct_answer = self.__dataset.test_y[i]
            answers[correct_answer] += 1
            if predicted_answer[0] == correct_answer:
                correct_answers += 1
                correctly_predicted[correct_answer] += 1
        accuracy = correct_answers / total_answers
        print(f"Accuracy: {accuracy}")
        print(f"Answers: {answers}")
        print(f"Predicted correctly: {correctly_predicted}")
