import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from uuid import uuid1
from copy import deepcopy
from typing import Callable
from scipy.interpolate import CubicSpline

from src.utils import dir_data, decode_object, encode_object, convolve, plot_hist, TimeStep
from src.sim_funcs import optimising_threshold, snr


class MeasurementStatisticsSimulator:
    time_format = '%Y_%m_%d-%H_%M_%S'

    def __init__(self, N: int, entangling_noise: Callable, entangling_noise_parameters: dict,
                 measurement_distribution_0: Callable, measurement_distribution_0_parameters: dict,
                 measurement_distribution_1: Callable, measurement_distribution_1_parameters: dict,
                 threshold_function: Callable = None, threshold_function_parameters: dict = None,
                 error_function: Callable = None, error_function_params: dict = None):
        """
        Principal object of the code. This object is used to run simulations of the space-time trade-off scheme
        described in https://arxiv.org/abs/2407.17342. Given the time dependent measurement statistics, for a single
        qubit measurement, CNOT gate noise in the form of a probability distribution and the number of qubits, this
        object can calculate the measurement statistics produced by our scheme and from that, a number of useful metrics
        for quantifying the quality of the scheme.

        :param N: Number of qubits.
        :param entangling_noise: Function which calculates the probability for the number of qubits in the excited state
                                after applying noisy CNOT gates across all qubits.
        :param entangling_noise_parameters: Input parameters for the entangling_noise function.
        :param measurement_distribution_0: Time dependent function which calculates the probability for the number of
                                          counts measured by a detector when a single qubit is in the |0> state.
        :param measurement_distribution_0_parameters: Input parameters for the measurement_distribution_0 function.
        :param measurement_distribution_1: Time dependent function which calculates the probability for the number of
                                          counts measured by a detector when a single qubit is in the |1> state.
        :param measurement_distribution_1_parameters: Input parameters for the measurement_distribution_1 function.
        :param threshold_function: Function which calculates the time dependent threshold value for discriminating
                                  between the |0> and |1> state.
        :param threshold_function_parameters: Input parameters for the threshold_function function.
        :param error_function: Function which calculates the figure of merit for the measurement statistics at each
                              point in time. e.g. signal to noise ratio or measurement infidelity.
        :param error_function_params: Input parameters for the error_function function.
        """

        self.id_ = None
        self.date = None

        self.N: int = N
        self.entangling_noise: Callable = entangling_noise
        self.entangling_noise_parameters: dict = entangling_noise_parameters
        self.measurement_distribution_0: Callable = measurement_distribution_0
        self.measurement_distribution_0_parameters: dict = measurement_distribution_0_parameters
        self.measurement_distribution_1: Callable = measurement_distribution_1
        self.measurement_distribution_1_parameters: dict = measurement_distribution_1_parameters
        self.threshold_function: str = threshold_function
        self.threshold_function_parameters: dict = threshold_function_parameters
        self.error_function: str = error_function
        self.error_function_params: dict = error_function_params

        self.k_max: int = None
        self.t_max: float = None
        self.result: list[TimeStep] = None

        if error_function is snr and threshold_function is optimising_threshold:
            raise TypeError("optimising_threshold does not work with snr")

    def _get_entangling_noise(self, q):
        if self.N == 1:
            # No entangling occurs for only 1 qubit
            return 1 if q == 1 else 0
        else:
            return self.entangling_noise(N=self.N, q=q, **self.entangling_noise_parameters)

    def _get_measurement_distribution_0(self, k, t):
        return self.measurement_distribution_0(k=k, t=t, **self.measurement_distribution_0_parameters)

    def _get_measurement_distribution_1(self, k, t):
        return self.measurement_distribution_1(k=k, t=t, **self.measurement_distribution_1_parameters)

    def _get_threshold_function(self, time_step: TimeStep):
        if self.threshold_function is optimising_threshold:
            # special threshold function which requires a copy of result to work as it scans measurement statistics.
            return self.threshold_function(time_step, self._get_error_function, **self.threshold_function_parameters)
        else:
            return self.threshold_function(t=time_step.t, **self.threshold_function_parameters)

    def _get_error_function(self, time_step: TimeStep):
        return self.error_function(time_step, **self.error_function_params)

    def get_time_and_error(self):
        e_list = []
        t_list = []

        for time_step in self.result:
            e_list.append(time_step.error)
            t_list.append(time_step.t)

        return e_list, t_list

    def get_time_from_error(self, error: float) -> float:
        """
        For a given error, find the time at which this error value was achieved. This function is more accurate with
        more TimeSteps in the results.

        :param error: error value

        :return: times at which error value occurs.
        """
        e_list, t_list = self.get_time_and_error()

        cs = CubicSpline(t_list, e_list)
        time = [float(t) for t in cs.solve(error) if t > min(t_list) and t < max(t_list)]

        return time

    def get_min_error(self) -> float:
        """
        Find the minimum error value. If error is snr then find max instead

        :return: minimum error
        """
        e_list, t_list = self.get_time_and_error()

        cs = CubicSpline(t_list, e_list)
        e_roots = [cs(t) for t in cs.derivative().roots() if t > min(t_list) and t < max(t_list)]
        if len(e_roots) == 0:
            e_min = max(e_list) if self.error_function is snr else min(e_list)
        else:
            e_min = float(min(e_roots))

        return e_min

    def find_k_max(self, t_max: float, k_upper: int, plot: bool = True, tol: float = 10e-8) -> None:
        """
        Find the maximum necessary value to run a simulation up to t_max. 

        :param t_max: maximum value of t to be simulated up to
        :param k_upper: upper_bound of k_max to start search from. For poisson 2*N*mu*t is a good first guess
        :param plot: plot the measurement statistics to check the correct value has been found
        :param tol: tolerance of k_max. k_max will be found when the probability of k is greater than tolerance
        """
        self.simulate([t_max], k_upper)
        # Start from the highest value of k and work down until the distribution increases above tol
        for i, p in enumerate(self.result[0].measurement_distribution_1[::-1]):
            if p > tol:
                break

        self.k_max = len(self.result[0].measurement_distribution_1) - i - 1
        self.result = []

        if plot:
            self.simulate([t_max])
            self.plot_measurement_statistics(0)
            plt.show()


    def simulate(self, t_list: list[float], k_max: int = None) -> None:
        """
        Simulate the measurement statistics for the defined parameters. We assume the statistics for an initial 0 state
        have no noise so we convolve the 0 state distribution N times whereas an initial state of 1 has some noise so
        the convolution is a combination of the 0 and 1 state distributions. Create a list [(time, statistics), ...]
        containing each of the time steps and the simulated statistics and store in self.result.

        :param t_list: time range to calculate distributions over
        :param k_max: maximum value of k needed to simulate the full statistics at the largest time. Calculate by
        default by calling find_k_max()
        """
        self.t_max = t_list[-1]
        if k_max is None:
            if self.k_max is None:
                raise ValueError("k_max has not been defined. Either pass it as an argument or call find_k_max()")
        else:
            self.k_max = k_max

        print(f"Simulating N={self.N}, k_max={self.k_max}, t={self.t_max}")

        self.result = []

        for t in tqdm(t_list):
            P0 = [self._get_measurement_distribution_0(k=k, t=t) for k in range(self.k_max)]
            P1 = [self._get_measurement_distribution_1(k=k, t=t) for k in range(self.k_max)]

            dist_0 = np.array(convolve(P0, P1, self.N, 0))

            dist_1 = np.zeros((self.k_max - 1) * (self.N + 1) + 1)
            for q in range(self.N + 1):
                dist_1 += self._get_entangling_noise(q) * np.array(convolve(P0, P1, self.N - q, q))

            self.result.append(TimeStep(t, dist_0, dist_1))

    def calculate_threshold(self) -> None:
        """
        Calculate the threshold value at each time step for the defined threshold function and add these to
        self.result. All clicks greater than or equal to the threshold value are defined as the 1 state.
        """
        if self.result is None:
            raise ValueError("You need to simulate before calculating the errors.")
        if self.threshold_function is None:
            raise ValueError("No threshold function has been defined.")

        for time_step in self.result:
            time_step.threshold_value = self._get_threshold_function(time_step)


    def calculate_error(self) -> None:
        """
        Calculate the error probabilities at each time step for the defined error function and add these to
        self.result.
        """
        if self.result is None:
            raise ValueError("You need to simulate before calculating the errors")

        for time_step in self.result:
            time_step.error = self._get_error_function(time_step)

    def plot_measurement_statistics(self, t_step: int, k_list: list[int] = None,
                                    distribution: int = None,
                                    kwargs_0: dict = {"label": "Distribution 0", "alpha": 0.5},
                                    kwargs_1: dict = {"label": "Distribution 1", "alpha": 0.5},
                                    title: str = None) -> None:
        """
        Plot the measurement statistic at the TimeStep (This is the index within the list, not the actual time value)

        :param t_step: index within list of results to plot
        :param k_list: range of k_values to plot
        :param: distribution: If 0 then only plot the 0 state distribution, if 1 then only plot the 1 state distribution
        and if None then plot both
        :param kwargs_0: kwargs for plot of distribution 0
        :param kwargs_1: kwargs for plot of distribution 1
        :param title: title of plot. Default will contain all relevant information on the plot
        """
        if k_list is None:
            k_list = np.arange(0, self.k_max+1)

        result = self.result[t_step]

        if distribution == 0 or distribution is None:
            plot_hist(lambda k: result.measurement_distribution_0[k], k_list, **kwargs_0)
        if distribution == 1 or distribution is None:
            plot_hist(lambda k: result.measurement_distribution_1[k], k_list, **kwargs_1)

        if (eta := result.threshold_value) is not None:
            plt.axvline(x = eta, color= "k", label = r'$\eta(t)$', linestyle='--')

        if title is None:
            ent_str = f"{self.entangling_noise.__name__}({self.entangling_noise_parameters})"
            meas_str_0 = f"{self.measurement_distribution_0.__name__}({self.measurement_distribution_0_parameters})"
            meas_str_1 = f"{self.measurement_distribution_1.__name__}({self.measurement_distribution_1_parameters})"
            title = (f"N={self.N}, t={result.t}\n"
                  f"entangling_noise={ent_str}\n"
                  f"measurement_distribution_0={meas_str_0}\n"
                  f"measurement_distribution_1={meas_str_1}")
            if (eta := result.threshold_value) is not None and self.threshold_function is not None:
                thr_str = f"{self.threshold_function.__name__}({self.threshold_function_parameters})"
                title += f"\nthreshold_function={thr_str}, eta={eta}"

        plt.title(title)
        plt.xlabel("Number of clicks (k)")
        plt.ylabel("Probability")
        plt.legend()

    def plot_entangling_noise(self, q_list: list[int],
                              title: str = None, **kwargs) -> None:
        """
        Plot probability of q out of self.N qubits in the 1 state for the entangling noise.

        :param q_list: list of number of qubits in the 1 state to plot
        :param title: title of plot. Default will contain all relevant information on the plot
        """
        plot_hist(self._get_entangling_noise, q_list, **kwargs)

        if title is None:
            ent_str = f"{self.entangling_noise.__name__}({self.entangling_noise_parameters})"
            title = (f"N={self.N}, entangling_noise={ent_str}")
        plt.title(title)

    def plot_error(self, title: str = None, t_step_max: float = None, t_transform: Callable = None,
                   e_transform: Callable = None, **kwargs) -> None:
        """
        Plot the error function for each time step

        :param title: title of plot. Default will contain all relevant information on the plot
        :param t_step_max: maximum time step to plot up to. If None then plot to the highest available time
        :param t_transform: function for transforming the time values. default to identity
        :param e_transform: function for transforming the error values. Default to identity
        """
        e_list, t_list = self.get_time_and_error()

        if t_step_max == None:
            t_step_max = len(t_list)

        if t_transform is not None:
            t_list = t_transform(np.array(t_list))
        if e_transform is not None:
            e_list = e_transform(np.array(e_list))

        plt.plot(t_list[:t_step_max], e_list[:t_step_max], **kwargs)

        if title is None:
            ent_str = f"{self.entangling_noise.__name__}({self.entangling_noise_parameters})"
            meas_str_0 = f"{self.measurement_distribution_0.__name__}({self.measurement_distribution_0_parameters})"
            meas_str_1 = f"{self.measurement_distribution_1.__name__}({self.measurement_distribution_1_parameters})"
            err_str = f"{self.error_function.__name__}({self.error_function_params})"
            title = (f"N={self.N}\n "
                     f"entangling_noise={ent_str}\n"
                     f"measurement_distribution_0={meas_str_0}\n"
                     f"measurement_distribution_1={meas_str_1}\n"
                     f"error_function={err_str}")
            if self.threshold_function is not None:
                thr_str = f"{self.threshold_function.__name__}({self.threshold_function_parameters})"
                title += f"\nthreshold_function={thr_str}"

        plt.title(title)
        plt.xlabel("time (t)")
        plt.ylabel("Error value")

    def save(self, file_name: str = None, dir: str = dir_data, mode: str = "a"):
        """ Save the data to a file. We don't only pickle the entire object as it isn't
            conducive to long term storage ie. if we ever want to update this class. JSON is better for this.
        """
        # Generate a new id and date for each save as any changes made should be stored as a unique circuit
        self.id_ = str(uuid1())
        self.date = datetime.now()

        if file_name is None:
            file_name = f"{self.date.strftime(self.time_format)}-{self.id_}.txt"

        save_dict = deepcopy(self.__dict__)

        save_dict["date"] = self.date.strftime(self.time_format)
        for key, item in save_dict.items():
            if isinstance(item, Callable):
                save_dict[key] = item.__name__

        with open(dir/file_name, mode) as f:
            f.write(str(save_dict))
            f.write("\n")
            f.write(str(encode_object(self)))
            f.write("\n")

    @classmethod
    def load(cls, file_name: str, dir: str = dir_data):
        """
        Create a MeasurementStatisticsSimulator object from a json string generated by the save method"""
        with open(dir/file_name, "r") as f:
            sim_pickle = f.readlines()[-1]

        return decode_object(sim_pickle)

if __name__ == "__main__":
    from src.sim_funcs import cascade_cnots_entangling_noise

    cascade_cnots_entangling_noise(10, 3, 0.005)
