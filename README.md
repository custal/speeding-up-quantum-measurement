This code is used to generate the data for the paper **Speeding up quantum measurement using space-time trade-off**: https://arxiv.org/abs/2407.17342

In the paper, we present a scheme for speeding up quantum measurement. The scheme builds on previous protocols that entangle the system to be measured with ancillary systems. In the idealised situation of perfect entangling operations and no decoherence, it gives an exact space-time trade-off meaning the readout speed increases linearly with the number of ancilla. We verify this scheme is robust against experimental imperfections through numerical modelling of gate noise and readout errors, and under certain circumstances our scheme can even lead to better than linear improvement in the speed of measurement with the number of systems measured. This hardware-agnostic approach is broadly applicable to a range of quantum technology platforms and offers a route to accelerate mid-circuit measurement as required for effective quantum error correction.

The principal object of the code is the MeasurementStatisticsSimulator object which can be found in src/core.py. This object is used to run simulations of the 
space-time trade-off scheme. Given the time dependent measurement statistics, for a single qubit
measurement, CNOT gate noise in the form of a probability distribution and the number of qubits, this object can calculate the measurement statistics
produced by our scheme and from that, a number of useful metrics for quantifying the quality of the scheme.

All the plots from the paper which present data from our scheme can be generated from scratch using the notebook in the jupyter folder.

If you have any questions about the code or the paper, please contact me at: christopher.corlett@bristol.ac.uk