import numpy as np
from PIL import Image, ImageOps

import Constants
from ANN import ANN
from Constants import MAX_PIXEL_VALUE, HIDDEN_LAYER_NEURONS, HIDDEN_LAYERS
from Functions import Functions
from OptimizationType import OptimizationType
from Regularization import RegularizationType

class MainApp:
    __done: bool

    def __init__(self):
        self.__done = False
        self.__options = {0: self.__exit, 1: self.__train, 2: self.__test_mnist, 3: self.__test_my_digits}

    def __exit(self):
        self.__done = True

    def __test_mnist(self):
        neural_network: ANN = ANN()
        neural_network.read_from_files(layer_count=HIDDEN_LAYERS + 1)
        neural_network.test()

    def __test_my_digits(self):
        neural_network: ANN = ANN()
        neural_network.read_from_files(layer_count=HIDDEN_LAYERS+1)
        for i in range(10):
            path: str = f"MyDigits/{i}.png"
            image: Image = Image.open(path)
            grayscale_image: Image = ImageOps.grayscale(image)

            gray_pixels = np.array(grayscale_image) / MAX_PIXEL_VALUE
            pixels: np.array = np.reshape(gray_pixels, (gray_pixels.shape[0] * gray_pixels.shape[1], 1))

            result = neural_network.predict_result(pixels)
            print(f"I have predicted {result} but the answer was {i}")

    def __train(self):
        optimization_type: str = input("The Optimization you want to use is: STOCHASTIC, MOMENTUM, RMS, ADA, ADAM\n>")
        optimization: OptimizationType = OptimizationType[optimization_type.strip().upper()]
        error_threshold: float = float(input("The error threshold you want to stop at is: \n>"))
        regularization_type: str = input("The regularization you want to use is: NONE, L1, L2\n>")
        regularization: RegularizationType = RegularizationType[regularization_type.strip().upper()]

        neural_network: ANN = ANN()
        neurons = [28 * 28]
        for _ in range(HIDDEN_LAYERS):
            neurons.append(HIDDEN_LAYER_NEURONS)
        neurons.append(10)
        activation_functions = [Functions.tanh for _ in range(HIDDEN_LAYERS + 1)]
        neural_network.initialise_neural_network(
            neurons_count=neurons,
            activation_functions=activation_functions,
            optimization=optimization,
            regularization=regularization
        )
        neural_network.train(
            threshold=error_threshold
        )
        print("Finished training the Neural Network")

    def __print_menu(self):
        print("0 to exit")
        print("1 to train the ANN")
        print("2 to test it on MNIST")
        print("3 to test it on my digits")
        print("4 to grid search")

    def run(self):
        while not self.__done:
            try:
                self.__print_menu()
                choice: int = int(input(">"))
                self.__options[choice]()
            except Exception as exception:
                print(exception)




if __name__ == '__main__':
    app = MainApp()
    app.run()

