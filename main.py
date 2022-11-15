# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle

import numpy as np
from PIL import Image, ImageOps

from ANN import ANN
from Constants import MAX_PIXEL_VALUE, HIDDEN_LAYER_NEURONS, HIDDEN_LAYERS
from Functions import Functions


def train_neural_network():
    # TODO: reg, opt
    neural_network: ANN = ANN()
    neurons = [28 * 28]
    for _ in range(HIDDEN_LAYERS):
        neurons.append(HIDDEN_LAYER_NEURONS)
    neurons.append(10)
    activation_functions = [Functions.sigmoid for _ in range(HIDDEN_LAYERS + 1)]
    neural_network.initialise_neural_network(neurons_count=neurons, activation_functions=activation_functions)
    neural_network.train()
    return neural_network

# def test_neural_network_user():
#     neural_network: NeuralNetwork = NeuralNetwork()
#     with open(MODEL_FILE, "rb") as file:
#         neural_network = pickle.load(file)
#     while True:
#         path: str = input("Path to image:")
#         try:
#             image: Image = Image.open(path)
#             grayscale_image: Image = ImageOps.grayscale(image)
#
#             gray_pixels = np.array(grayscale_image) / MAX_PIXEL_VALUE
#             pixels: np.array = np.reshape(gray_pixels, (gray_pixels.shape[0] * gray_pixels.shape[1], 1))
#
#             result: tuple[int, float] = neural_network.predict_result(pixels=pixels)
#             print(f"The digit chosen is {result[0]} with a chance of {round(result[1], 4) * 100}%")
#         except Exception as exception:
#             print(exception)


def test_neural_network():
    neural_network: ANN = ANN()
    neural_network.read_from_files([
        ("model/w1.nn", "model/b1.nn", "model/f1.nn"),
        ("model/w2.nn", "model/b2.nn", "model/f2.nn"),
        ("model/wo.nn", "model/bo.nn", "model/fo.nn")
    ])
    neural_network.test()
    # neural_network.test_on_test_data()


def test_neural_network_user():
    neural_network: ANN = ANN()
    with open("model.ann", "rb") as file:
        neural_network = pickle.load(file)
    while True:
        path: str = input("Path to image:")
        try:
            image: Image = Image.open(path)
            grayscale_image: Image = ImageOps.grayscale(image)

            gray_pixels = np.array(grayscale_image) / MAX_PIXEL_VALUE
            pixels: np.array = np.reshape(gray_pixels, (gray_pixels.shape[0] * gray_pixels.shape[1], 1))

            result = neural_network.predict_result(pixels)
            print(f"The digit chosen is {result}")
        except Exception as exception:
            print(exception)


# test_neural_network()


if __name__ == '__main__':
    train_neural_network()
    # test_neural_network()
    # test_neural_network()
    # test_neural_network_user()
    # ann.initialise_neural_network(neurons_count=[28 * 28, 16, 16, 10], activation_functions=[Sigmoid() for _ in range(3)])
    # ann.train()
    # neural_network = train_neural_network()
    # neural_network.test_on_test_data()
    # test_neural_network_user()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
