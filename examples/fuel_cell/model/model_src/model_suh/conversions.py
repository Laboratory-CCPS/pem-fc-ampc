import numpy as np


def celsius2kelvin(T_C):
    return 273.15 + T_C


def kelvin2celsius(T_K):
    return T_K - 273.15


def rpm2rad(rpm):
    return rpm / 60 * (2 * np.pi)


def rad2rpm(rad):
    return rad * 60 / (2 * np.pi)
