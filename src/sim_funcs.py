"""
Functions to initialise MeasurementStatisticsSimulator
"""

from scipy.stats import poisson
from scipy.integrate import quad
import numpy as np
from src.utils import top_hat, mean, var, TimeStep
from typing import Callable


def cascade_cnots_entangling_noise(N: int, q: int, p: float) -> float:
    """ Function returns the probability of q 1 state qubits given N inputs for a cascade entanglement protocol.
    See equations (10) and (11) in the Supplementary material.

    :param p: probability of CNOT failing
    :param N: total number of qubits attempting to entangle
    :param q: number of qubits entangled

    :return: probability of entangling q qubits.
    """
    if N == 1:
        return 1 if q == 1 else 0
    elif (M := int(N / 2)) == N / 2:
        return (1 - p) ** (q + 1) * p ** 2 * (
                    (q + 1) * top_hat(0, M - 2, q) + (2 * M - 3 - q) * top_hat(M - 1, 2 * (M - 2), q)) + (1 - p) ** (
                           q - 1) * top_hat(2 * M, 2 * M, q) + 2 * (1 - p) ** q * p * top_hat(M, 2 * M - 2,
                                                                                              q) + p * top_hat(0, 0, q)
    else:
        return (1 - p) ** (q + 1) * p ** 2 * (
                    (q + 1) * top_hat(0, M - 2, q) + (2 * M - 3 - q) * top_hat(M - 1, 2 * (M - 2), q) +
                    top_hat(M - 1, 2 * M - 3, q)) + (1 - p) ** (q - 1) * top_hat(2 * M + 1, 2 * M + 1, q) + \
               (1 - p) ** q * p * (2 * top_hat(M + 1, 2 * M - 1,q) + top_hat(M, M, q)) + p * top_hat(0, 0, q)

def flat_cnots_entangling_noise(N: int, q: int, p: float) -> float:
    """
    Function returns the probability of q 1 state qubits given N inputs for a flat entanglement protocol.
    See equation (9) in the Supplementary Material.

    :param p: probability of CNOT failing
    :param N: total number of qubits attempting to entangle
    :param q: number of qubits entangled

    :return: probability of entangling q qubits.
    """
    return (1 - p) ** q * p * top_hat(0, N - 2, q) + (1 - p) ** (q - 1) * top_hat(N, N, q)

def perfect_entangling_noise(N: int, q: int) -> float:
    """
    Function returns the probability of q 1 state qubits given N inputs for a perfect entanglement protocol

    :param N: total number of qubits attempting to entangle
    :param q: number of qubits entangled

    :return: probability of entangling q qubits.
    """
    if q == N:
        return 1
    else:
        return 0

def worst_entangling_noise(N: int, q: int, p: float) -> float:
    """
    Function returns the probability of q 1 state qubits given N inputs for the worst entanglement protocol

    :param N: total number of qubits attempting to entangle
    :param q: number of qubits entangled
    :param p: probability of CNOT failing

    :return: probability of entangling q qubits.
    """
    if q == N:
        return (1-p)**(N-1)
    elif q == 0:
        return 1 - (1-p)**(N-1)
    else:
        return 0

def perfect_poisson(k: int, t: float, mu: float) -> float:
    """
    Return the probability of k clicks for a poisson distribution with mean mu * t. See equation (2)

    :param k: number of clicks
    :param t: time of measurement
    :param mu: emission rate

    :return: probability of k clicks at time t
    """
    return poisson.pmf(k, mu * t)

def decaying_poisson(k: int, t: float, mu0: float, mu1: float, lam: float) -> float:
    """
    Return the probability of k clicks for a poisson distribution at time t decaying from an emission rate of mu1 to
    mu0 at rate lam.
    See equation (6) in the Appendix

    :param k: number of clicks
    :param t: time of measurement
    :param mu0: emission rate of ground state
    :param mu1: emission rate of excited state
    :param lam: rate of decay

    :return: probability of k clicks at time t
    """

    def integrand(t_prime: float, k: int, t: float, mu0: float, mu1: float, lam: float):
        return lam * np.exp(-lam * t_prime) * poisson.pmf(k, (mu1 - mu0) * t_prime + mu0 * t)

    result = quad(lambda t_prime: integrand(t_prime, k, t, mu0, mu1, lam), 0, t)

    return result[0] + np.exp(-lam * t) * poisson.pmf(k, mu1 * t)

def linear_threshold(t: float, eta: float) -> float:
    """
    Calculate the threshold value for a linear map

    :param t: time
    :param eta: scaling

    :return: threshold value
    """
    return int(eta * t)

def decaying_threshold(t: float, eta: float, mu: float, lam: float) -> float:
    """
    Calculate the threshold value for a decaying map

    :param t: time
    :param eta: scaling
    :param mu: value to decay towards
    :param lam: decay rate

    :return: threshold value
    """
    return int((mu-(mu-eta)*np.exp(-lam*t)) * t)

def optimising_threshold(time_step: TimeStep, error_function: Callable, mu0: float, mu1: float) -> int:
    """
    Find the optimal threshold value by checking all threshold values between mu0*t and mu1*t and choosing the lowest

    :param time_step: results for a TimeStep from MeasurementStatisticsSimulator so results can be accessed and scanned
    :param error_function: error function to optimise over
    :param mu0: lower bound of search
    :param mu1: upper bound of search

    :return: optimal threshold value
    """
    t = time_step.t

    lowest_error = np.inf
    for eta in range(int(mu0*t), int(np.ceil(mu1*t))+1):
        time_step.threshold_value = eta
        error = error_function(time_step)
        if error < lowest_error:
            lowest_error = error
            best_eta = eta

    return best_eta


def average_error(time_step: TimeStep) -> float:
    """
    Calculate the average measurement error.

    :param time_step: TimeStep of simulation

    :return: average error
    """
    if (eta := time_step.threshold_value) is None:
        raise KeyError("Could not find threshold_value. Make sure calculate_threshold has been run.")

    e0 = sum(time_step.measurement_distribution_0[eta:-1])
    e1 = sum(time_step.measurement_distribution_1[0:eta])

    return (e0 + e1)/2

def snr(time_step: TimeStep) -> float:
    """
    Calculate the snr. See equation (1)

    :param time_step: TimeStep of simulation

    :return: snr
    """
    dist_0 = np.array(time_step.measurement_distribution_0)
    dist_1 = np.array(time_step.measurement_distribution_1)

    outcomes = np.arange(len(dist_0))
    a0 = mean(dist_0, outcomes)
    a1 = mean(dist_1, outcomes)
    v0 = var(dist_0, outcomes)
    v1 = var(dist_1, outcomes)

    return 2 * abs(a0 - a1) / (np.sqrt(v0) + np.sqrt(v1))