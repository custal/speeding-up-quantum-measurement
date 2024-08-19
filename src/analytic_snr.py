import numpy as np
from typing import Callable
from scipy.special import gamma, gammainc
from src.utils import mean, var
from src.sim_funcs import cascade_cnots_entangling_noise


def analytic_snr(E0: Callable, E1: Callable, V0: Callable, V1: Callable, ET: Callable, VT: Callable, N: int,
                 t: float) -> float:
    """
    Function for calculating the snr, of a probability distribution, analytically at time t and for N qubits, using the
    equation defined in the supplementary material (20).

    :param E0: Function for expectation of single qubit in state |0>. Takes t as input.
    :param E1: Function for expectation of single qubit in state |1>. Takes t as input.
    :param V0: Function for variance of single qubit in state |0>. Takes t as input.
    :param V1: Function for variance of single qubit in state |1>. Takes t as input.
    :param ET: Function for expectation of CNOT noise distribution normalised by N. Takes N as input.
    :param VT: Function for variance of CNOT noise distribution normalised by N. Takes N as input.
    :param N: Number of qubits
    :param t: time of measurement

    :return: snr for the specified parameters
    """
    return np.sqrt(N) * 2*(E1(t)-E0(t))*ET(N) / (np.sqrt(V0(t)) + np.sqrt(V0(t) + (V1(t)-V0(t))*ET(N) +
                                                                          (E1(t)-E0(t))**2 * VT(N)))

def expectation_0(mu0):
    """See equation (2) for N=1"""
    return lambda t: mu0*t

def expectation_1(mu0, mu1, lam):
    """See equation (6) in the Appendix"""
    if lam == 0:
        return lambda t:  mu1 * t
    else:
        return lambda t: gamma(2)*gammainc(2, lam*t)*(mu1 - mu0) / lam + (mu1 - mu0)*t*np.exp(-lam*t) + mu0*t

def expectation_cnot(p, prob_dist: Callable = cascade_cnots_entangling_noise):
    """prob_dist is the probability distribution for the probability of getting q excited qubits. Haven't solved
    analytic expression yet"""
    return lambda N: mean([prob_dist(N, q, p) for q in range(N+1)], np.arange(N+1))/N

def variance_cnot(p, prob_dist: Callable = cascade_cnots_entangling_noise):
    """prob_dist is the probability distribution for the probability of getting q excited qubits. Haven't solved
    analytic expression yet"""
    return lambda N: var([prob_dist(N, q, p) for q in range(N+1)], np.arange(N+1))/N