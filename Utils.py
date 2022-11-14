import numpy as np
from keras.datasets import mnist

from Constants import MAX_PIXEL_VALUE, MAX_WEIGHT_VALUE, MIN_WEIGHT_VALUE, RANDOM_DISTRIBUTION, MEAN, STANDARD_DEVIATION


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


def generate_random_matrix(lines: int,
                           columns: int,
                           distribution: str = RANDOM_DISTRIBUTION,
                           *args) -> np.array:
    if distribution == "NORMAL":
        mean: float = MEAN if len(args) == 0 else float(args[0])
        standard_deviation: float = STANDARD_DEVIATION if len(args) <= 1 else float(args[1])
        return np.random.normal(mean, standard_deviation, size=(lines, columns))
    if distribution == "UNIFORM":
        min_value: float = MIN_WEIGHT_VALUE if len(args) == 0 else float(args[0])
        max_value: float = MAX_WEIGHT_VALUE if len(args) <= 1 else float(args[1])
        return np.random.uniform(min_value, max_value, size=(lines, columns))
    raise Exception("Unknown distribution")



