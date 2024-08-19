import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Iterable
from pathlib import Path
import pickle
import base64
from dataclasses import dataclass

dir_data = Path(__file__).parents[1]/"data"
dir_images = Path(__file__).parents[1]/"images"

dir_data.mkdir(exist_ok=True)
dir_images.mkdir(exist_ok=True)

@dataclass
class TimeStep:
    """Data needed at each time step in a simulation"""
    t: float
    measurement_distribution_0: list[float]
    measurement_distribution_1: list[float]
    threshold_value: float = None
    error: float = None

def encode_object(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")

def decode_object(obj):
    return pickle.loads(base64.b64decode(obj))

def top_hat(lower, upper, x):
    if x >= lower and x <= upper:
        return 1
    else:
        return 0

def mean(pmf, outcomes):
    return sum(pmf * outcomes)

def var(pmf, outcomes):
    return mean(pmf, outcomes ** 2) - mean(pmf, outcomes) ** 2

def plot_hist(dist, k_list, **kwargs):
    plt.hist(k_list, np.array(k_list)-0.5, weights=[dist(k) for k in k_list], **kwargs)


def convolve(P0, P1, N0, N1):
    """ Convolve the distribution P0 N0 times with the distribution P1 N1 times. The distributions must be in the form
    of a list

    P0 (List[float]): Probability distribution
    P1 (List[float]): Probability distribution
    N0 (int): Number of times to convolve P0
    N1 (int): Number of times to convolve P1

    return (List[float]): convolution of P0 and P1. Will have dimensions len(P0)+len(P1)-1
    """
    # Convolution kernel
    dist = [0] * len(P0)
    dist[0] = 1

    for i in range(N0):
        dist = np.convolve(dist, P0)
    for i in range(N1):
        dist = np.convolve(dist, P1)

    return dist

def plot_constant_error(simulator_list: list, error: float, normalised: bool = False, inverse: bool = False,
                        label: str = None, color: str = None, marker: str = None) -> None:
    """
    Plot lines of constant error for time of measurement to achieve the error against qubit number

    :param simulator_list: List is MeasurementStatisticsSimulator objects
    :param error: error value to plot lines from
    :param normalised: If True then normalise all points using the first point
    :param inverse: if True then plot the inverse of the points
    :param label: Label for points on the legend
    :param color: point color
    :param marker: point marker
    """
    # Get lowest error for each qubit
    t_list = np.array([])
    q_list = []

    for simulator in simulator_list:
        t_err = simulator.get_time_from_error(error)
        if len(t_err) != 0:
            t_list = np.append(t_list, min(t_err))
            q_list.append(simulator.N)

    if normalised:
        t_list = t_list / t_list[0]
    if inverse:
        t_list = t_list ** -1

    plt.scatter(q_list, t_list, label=label, marker=marker, color=color)

def plot_lowest_errors(simulator_list: list) -> None:
    """
    Plot the lowest error achieved against number of qubits for each simulator in simulator_list

    :param simulator_list: List is MeasurementStatisticsSimulator objects
    """
    lowest_error_list = []
    qubit_num_list = []

    # Get lowest error for each qubit
    for simulator in simulator_list:
        lowest_error_list.append(simulator.get_min_error())
        qubit_num_list.append(simulator.N)

    # Plot lowest error for each qubit
    plt.scatter(qubit_num_list, lowest_error_list, marker="x", color="r")
