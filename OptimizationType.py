from enum import Enum

import numpy as np

from Constants import MOMENTUM_BETA, RMSPROP_BETA_2, ADAM_BETA_1, ADAM_BETA_2


class GradientDescent:
    def __init__(self, shape):
        pass

    def make_step(self, dW: np.array) -> np.array:
        pass


class StochasticGradientDescent(GradientDescent):
    def __init__(self, shape):
        super().__init__(shape)

    def make_step(self, dW: np.array) -> np.array:
        return dW


class MomentumGradientDescent(GradientDescent):
    __v_dW: np.array

    def __init__(self, shape):
        super().__init__(shape)
        self.__v_dW = np.zeros(shape=shape)

    def make_step(self, dW: np.array) -> np.array:
        self.__v_dW = MOMENTUM_BETA * self.__v_dW + (1 - MOMENTUM_BETA) * dW
        return self.__v_dW


class AdaGrad(GradientDescent):
    __cache: np.array

    def __init__(self, shape):
        super().__init__(shape)
        self.__cache = np.zeros(shape=shape)

    def make_step(self, dW: np.array) -> np.array:
        self.__cache += dW ** 2
        return dW / (self.__cache ** (1 / 2) + np.exp(-7))


class RMSProp(GradientDescent):
    __cache: np.array

    def __init__(self, shape):
        super().__init__(shape)
        self.__cache = np.zeros(shape=shape)

    def make_step(self, dW: np.array) -> np.array:
        self.__cache = RMSPROP_BETA_2 * self.__cache + (1 - RMSPROP_BETA_2) * (dW ** 2)
        return dW / (self.__cache ** (1 / 2) + np.exp(-7))


class Adam(GradientDescent):
    __m: np.array
    __v: np.array

    def __init__(self, shape):
        super().__init__(shape)
        self.__m = np.zeros(shape=shape)
        self.__v = np.zeros(shape=shape)

    def make_step(self, dW: np.array):
        self.__m = ADAM_BETA_1 * self.__m + (1 - ADAM_BETA_1) * dW
        self.__v = ADAM_BETA_2 * self.__v + (1 - ADAM_BETA_2) * (dW ** 2)
        return self.__m / (self.__v ** (1 / 2) + np.exp(-7))


class OptimizationType(Enum):
    STOCHASTIC = 1
    MOMENTUM = 2
    ADA = 3
    RMS = 4
    ADAM = 5


class OptimizationFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_optimization(optimization: OptimizationType, shape: tuple[int]) -> GradientDescent:
        if optimization == OptimizationType.STOCHASTIC:
            return StochasticGradientDescent(shape)
        if optimization == OptimizationType.MOMENTUM:
            return MomentumGradientDescent(shape)
        if optimization == OptimizationType.ADA:
            return AdaGrad(shape)
        if optimization == OptimizationType.RMS:
            return RMSProp(shape)
        if optimization == OptimizationType.ADAM:
            return Adam(shape)
