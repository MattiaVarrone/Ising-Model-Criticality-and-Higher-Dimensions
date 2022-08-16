import numpy as np
from scipy.integrate import quad


def E(T):
    k1_0 = 2 * np.sinh(2 / T) * (np.cosh(2 / T))**(-2)
    k1 = k1_0 ** 2
    k2 = 2 * np.tanh(2 / T) ** 2 - 1

    def ell_1(x):
        return (1 - k1 * np.sin(x) ** 2) ** (-0.5)
    K1 = quad(ell_1, 0, np.pi / 2)[0]

    E = -(np.tanh(2 / T))**(-1) * (1 + 2/np.pi * k2 * K1)
    return E

def C(T):
    k1_0 = 2 * np.sinh(2 / T) * (np.cosh(2 / T))**(-2)
    k1 = k1_0 ** 2
    k2 = 2 * np.tanh(2 / T) ** 2 - 1

    def ell_1(x):
        return (1 - k1 * np.sin(x) ** 2) ** (-0.5)

    def ell_2(x):
        return (1 - k1 * np.sin(x) ** 2) ** (+0.5)

    K1 = quad(ell_1, 0, np.pi / 2)[0]
    E1 = quad(ell_2, 0, np.pi / 2)[0]

    C = (2 / np.pi) * (T * np.tanh(2 / T)) ** (-2) * (2 * K1 - 2 * E1 - (1 - k2) * (np.pi / 2 + k2 * K1))
    return C

