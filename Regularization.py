from enum import Enum

import numpy as np


class Regularization:
    def __init__(self):
        pass

    def forward(self, W: np.array) -> float:
        pass

    def backward(self, W: np.array) -> np.array:
        pass


class NoRegularization(Regularization):
    def __init__(self):
        super().__init__()

    def forward(self, W: np.array) -> float:
        return 0

    def backward(self, W: np.array) -> np.array:
        return np.zeros_like(W)


class L1Regularization(Regularization):
    def __init__(self):
        super().__init__()

    def forward(self, W: np.array) -> float:
        return float(np.sum(np.abs(W)))

    def backward(self, W: np.array) -> np.array:
        dW: np.array = np.zeros_like(W)
        dW[W > 0] = 1
        dW[W < 0] = -1
        return dW


class L2Regularization(Regularization):
    def __init__(self):
        super().__init__()

    def forward(self, W: np.array) -> float:
        return float(np.sum(W ** 2))

    def backward(self, W: np.array) -> np.array:
        return 2 * W


class RegularizationType(Enum):
    NONE = 1
    L1 = 2
    L2 = 3


class RegularizationFactory:
    __no_regularization: NoRegularization
    __l1_regularization: L1Regularization
    __l2_regularization: L2Regularization

    def __init__(self):
        self.__no_regularization = NoRegularization()
        self.__l1_regularization = L1Regularization()
        self.__l2_regularization = L2Regularization()

    def create_regularization(self, regularization: RegularizationType) -> Regularization:
        if regularization == RegularizationType.NONE:
            return self.__no_regularization
        if regularization == RegularizationType.L1:
            return self.__l1_regularization
        if regularization == RegularizationType.L2:
            return self.__l2_regularization


factory = RegularizationFactory()
