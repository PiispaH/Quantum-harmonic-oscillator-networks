import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import colormap as cm
import timeit
import networkx as nx
# from random import randint
from json import load
from matplotlib.animation import PillowWriter
# import matplotlib;matplotlib.use("TkAgg")

"""                                                              """
"""          Setting the constants and parameter values          """
"""                                                              """

# Reading the config file
with open("config.json") as file:
    constants = load(file)

N = constants["N"]                                      # Number of oscillators
T = constants["T"]                                      # Temperature
INITIAL_Q = constants["INITIAL_Q"]                      # Values of Q in the initial state
INITIAL_P = constants["INITIAL_P"]                      # Values of P in the initial state
INITIAL_STATE_VECTOR = np.array(INITIAL_Q + INITIAL_P)  # Vector of first moments
SQUEEZING_PARAMETERS = np.array(constants["r"])         # The squeezing parameters
SQUEEZING_ANGLES = np.array(constants["phi"])           # The squeezing angles

MIN_X_AND_Y = constants["MIN_X_AND_Y"]                  # Minimum value for x and y in plots
MAX_X_AND_Y = constants["MAX_X_AND_Y"]                  # Maximum value for x and y in plots
RESOLUTION = constants["RESOLUTION"]                    # Resolution of the plots
DURATION = constants["DURATION"]                        # The duration of the simulation
TIME_STEP = constants["TIME_STEP"]                      # Time step of simulation
SAVE = constants["SAVE"]                                # Whether to save the animations and plots or not
STEPS = int(DURATION / TIME_STEP)                       # How many steps in whole simulation
TIMES = np.linspace(0, STEPS, STEPS + 1) * TIME_STEP    # Instances of time, where values are calculated
FREQUENCIES = np.ones(N)                                # The frequencies of the oscillators

# Defining a NxN zero matrix to help construct the other matrices
zero_block = np.zeros((N, N))

# Defining the symplectic J matrix
J = np.block([[zero_block, np.identity(N)],
              [-1 * np.identity(N), zero_block]])

# For testing with many oscillators
second_neighbor = False
chain = True
if N == 0:
    N = 10
    INITIAL_STATE_VECTOR = np.concatenate((1 * np.ones(0), np.zeros(2 * N - 0)))
    SQUEEZING_PARAMETERS = np.concatenate((np.zeros(1), -1 * np.ones(N - 1)))
    # SQUEEZING_PARAMETERS = np.concatenate((0.2 * np.ones(1), np.zeros(N - 1)))
    # SQUEEZING_PARAMETERS = -1 * np.ones(N)
    SQUEEZING_ANGLES = np.zeros(N)
    FREQUENCIES = np.ones(N)
    zero_block = np.zeros((N, N))
    J = np.block([[zero_block, np.identity(N)],
                  [-1 * np.identity(N), zero_block]])


def thermal_excitations(cov: np.ndarray) -> float:
    """
    Calculates the amount of thermal excitations in the system

    :param cov: covariance matrix of the system
    :return: amount of thermal excitations
    """
    th_excitations = np.sqrt(np.linalg.det(cov)) - 0.5
    return th_excitations


def initial_cov_matrix(freqs: np.ndarray, squeezing_parameters: np.ndarray,
                       phis: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix of the whole network at t=0.

    :param freqs: frequencies
    :param squeezing_parameters: Contains all the squeezing parameters of the network at t=0
    :param phis: Contains all the squeezing angles of the network at t=0
    :return cov_X_initial: Covariance matrix of the initial state
    """

    # Creates a template for the covariance matrix
    cov_X_initial = np.zeros((2 * N, 2 * N))

    # Checking the indices of vacuum states
    vacuum_indices = np.where(SQUEEZING_PARAMETERS == -1)[0]

    # Checks which oscillators have squeezed states and which don't have
    squeezed_indices = np.where(SQUEEZING_PARAMETERS > 0)[0]

    # Iterates through each state
    for i in range(N):
        phi = phis[i]
        r = squeezing_parameters[i]
        freq = freqs[i]

        # Defining the symplectic squeezing matrix
        symp_squeezing_matrix = np.zeros((2, 2))
        symp_squeezing_matrix[0][0] = np.cosh(r) + np.sinh(r) * np.cos(phi)
        symp_squeezing_matrix[1][1] = np.cosh(r) - np.sinh(r) * np.cos(phi)
        symp_squeezing_matrix[1][0] = freq * np.sinh(r) * np.sin(phi)
        symp_squeezing_matrix[0][1] = (1 / freq) * np.sinh(r) * np.sin(phi)

        # If simulating a vacuum state or a squeezed state use hand calculated limit for the general
        # case of thermal states. Else use the general thermal state version with temperature T
        if T == 0 or i in np.concatenate((vacuum_indices, squeezed_indices)):
            thermal_expectation_term = 0.5
        else:
            thermal_expectation_term = 1 / (np.exp(freq / T) - 1) + 0.5

        cov_osc_i = np.diag(np.array([thermal_expectation_term / freq, thermal_expectation_term * freq]))

        # Perform squeezing if the state is squeezed
        if i in squeezed_indices:
            cov_osc_i = symp_squeezing_matrix @ cov_osc_i @ symp_squeezing_matrix.transpose()

        cov_X_initial[i][i] = cov_osc_i[0][0]
        cov_X_initial[i + N][i + N] = cov_osc_i[1][1]
        cov_X_initial[i + N][i] = cov_osc_i[1][0]
        cov_X_initial[i][i + N] = cov_osc_i[0][1]

    return cov_X_initial


def new_cov_calculation(S: np.ndarray, cov_X: np.ndarray):
    """
    Calculates the new covariance matrix at moment t in time.

    :param S: Symplectic matrix calculated at moment t in time
    :param cov_X: Initial state covariance matrix
    :return: diagonal: the variances as an vector, new_cov_X: the new covariance matrix
    that describes the network at moment t as an array.
    :rtype: (np.ndarray, np.ndarray)
    """

    new_cov_X = S @ cov_X @ S.transpose()
    diagonal = np.diagonal(new_cov_X)
    return diagonal, new_cov_X


def save_into_dict(t: float, storage: dict, non: np.ndarray,
                   inte: np.ndarray, old: np.ndarray) -> None:
    """
    Saves the passed values into a corresponding dict

    :param t: Moment in time
    :param storage: The dict into where the data is saved
    :param non: The value in non interacting base
    :param inte: The value in interacting base
    :param old: The value in the old base
    :return: None
    """
    # Iterates through every oscillator in the system
    for i in range(N):
        # Need to initialize the lists and dicts if t=0
        if t == 0:
            storage[f"{i}"] = {'Q': [(non[i], inte[i], old[i])],
                               'P': [(non[i + N], inte[i + N], old[i + N])]}

        # If dicts and lists are already there, just append the new values to the lists
        else:
            storage[f"{i}"]['Q'].append((non[i], inte[i], old[i]))
            storage[f"{i}"]['P'].append((non[i + N], inte[i + N], old[i + N]))


def cosine(frequency: np.ndarray, t: float) -> np.ndarray:
    """
    Computes cos(Ωt) or cos(ωt) depending on which frequencies are
    used.

    :param frequency: Frequencies of the oscillators
    :param t: Time
    :return: Cosines of the frequencies multiplied by t
    """
    return np.cos(frequency * t)


# Vectorizes the function for convenience
cosine_vector = np.vectorize(cosine)


def sine(frequency: np.ndarray, t: float) -> np.ndarray:
    """
    Computes sin(Ωt) or sin(ωt) depending on which frequencies are
    used.
    :param frequency: Frequencies of the oscillators
    :param t: Time
    :return: Cosines of the frequencies multiplied by t
    """
    return np.sin(frequency * t)


# Vectorizes the function for convenience
sine_vector = np.vectorize(sine)


def network_initialize(plot=False) -> nx.Graph:
    """
    Creates a network of N nodes with specified links and weights

    :param plot: Whether to plot the graph or not
    :return: A networkx Graph -object
    """

    # If only one oscillator is simulated create a graph
    # with only one node and no links
    if N == 1:
        G = nx.newman_watts_strogatz_graph(N, 1, 0.0)

    # If more than one oscillator are simulated, first create
    # a ring of nodes each with one edge to either side
    else:
        G = nx.newman_watts_strogatz_graph(N, 2, 0.0)

    # Each edge has a weight of 1
    for u, v in G.edges():
        G.edges[u, v]['weight'] = 1

    # If every node is also connected to its 2nd neighbours
    # create new edges with specified weights
    if second_neighbor:
        second_neighbors = []
        for u, v in G.edges():
            if u < N - 2:
                second_neighbors.append((u, u + 2))
        G.add_edges_from(second_neighbors, weight=0.02)

    # If network is a chain, remove the "last" edge
    if N > 1 and chain:
        G.remove_edge(0, N - 1)

    # Draws the network with edge weights and node labels in a circle
    if plot and N > 1:
        nx.draw_networkx_labels(G, pos=nx.circular_layout(G))
        nx.draw_circular(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G),
                                     edge_labels=labels)

    return G


def matrix_K(G: nx.Graph, frequencies: np.ndarray):
    """
    Calculates the matrix A that describes the network of quantum mechanical harmonic oscillators.

    :param G: The network
    :return: K: matrix of eigenvectors of matrix A, eigenfrequencies: vector of eigenfrequencies
    :rtype: (np.ndarray, np.ndarray)
    """

    # Adjacency matrix
    V = nx.adjacency_matrix(G).toarray()

    # Degree matrix
    D = np.diag([val for (node, val) in sorted(G.degree(), key=lambda pair: pair[0])])

    # Laplacian matrix
    L = D - V
    print("\nL:")
    print(L)

    # Matrix A
    A = np.diag(frequencies ** 2) * 0.5 + 0.5 * L
    print("\nMatrix A:")
    print(A)

    # Calculates the matrix K and eigenfrequencies
    eigenvalues, K = np.linalg.eigh(A)
    eigenfrequencies = np.sqrt(eigenvalues * 2)

    print("\nMatrix K:")
    print(K)

    # A check is done to see that the diagonalizing has been done properly
    if not sanity_check_2(eigenvalues, A, K):
        raise Exception("Diagonalizing of matrix A failed")

    return K, eigenfrequencies


# Creates the symplectic matrix for a specific moment in time
def symplectic(t: float, K: np.ndarray, freqs: np.ndarray, base: str) -> np.ndarray:
    """
    Calculates the symplectic matrix in either of three bases, noninteracting, interacting or old.

    :param t: Moment in time
    :param K: Eigenvectors of matrix A in a matrix form
    :param freqs: Oscillators frequencies
    :param base: In what base the symplectic matrix is calculated in
    :return: S: the symplectic matrix
    """

    # Calculates the NxN diagonal matrices consisting of sin(Ωt) and cos(Ωt)
    block_cos = np.diag(cosine_vector(freqs, t))
    block_sin = np.diag(sine_vector(freqs, t))

    # If calculating the symplectic matrix for the old base, use interacting frequencies
    # and create diagonal vectors of the frequencies and another one of their reciprocals.
    # If one of the other bases, does the same with eigenfrequencies
    if base == "old":
        freq_diag = np.diag(FREQUENCIES)
        inverse_freq_diag = np.linalg.inv(freq_diag)
    else:
        freq_diag = np.diag(freqs)
        inverse_freq_diag = np.linalg.inv(freq_diag)

    # If base is noninteracting create standard S
    if base == "non_interacting":
        S = np.block([[block_cos, inverse_freq_diag * block_sin],
                      [-freq_diag * block_sin, block_cos]])

    # If interacting base, multiply each element from left with matrix K
    elif base == "interacting":
        S = np.block([[K @ block_cos, K @ (inverse_freq_diag * block_sin)],
                      [K @ (-freq_diag * block_sin), K @ block_cos]])

    # If old base calculate S with interacting frequencies and multibly from left and right (transpose)
    # with a diagonal matrix containing transposes of K
    elif base == "old":
        nonint_to_int = np.block([[K.transpose(), zero_block],
                                  [zero_block, K.transpose()]])
        S = np.block([[block_cos, inverse_freq_diag * block_sin],
                      [-freq_diag * block_sin, block_cos]])
        S = nonint_to_int.transpose() @ S @ nonint_to_int

    # Raise an ValueError if the given base is not one of th three possible
    else:
        raise ValueError("Function symplectic() argument 'kind' invalid")

    return S


def sanity_check_1(S1: np.ndarray, S2: np.ndarray, S3: np.ndarray) -> bool:
    """
    Checks that all of the calculated symplectic matrices are indeed still symplectic.
    Raises an ValueError if at least one of the matrices is not symplectic.

    :param S1: Symplectic matrix in base 1
    :param S2: Symplectic matrix in base 2
    :param S3: Symplectic matrix in base 3
    :return: True if all of the matrices are symplectic
    """
    new_J_matrix = S1 @ J @ np.transpose(S1)
    if not np.allclose(J, new_J_matrix):
        raise ValueError(f"The matrix non is not symplectic anymore")
    new_J_matrix = S2 @ J @ np.transpose(S2)
    if not np.allclose(J, new_J_matrix):
        raise ValueError(f"The matrix int is not symplectic anymore")
    new_J_matrix = S3 @ J @ np.transpose(S3)
    if not np.allclose(J, new_J_matrix):
        raise ValueError(f"The matrix old is not symplectic anymore")

    return True


def sanity_check_2(eigenvalues, A: np.ndarray, K: np.ndarray) -> bool:
    """
    Checks if the Hamiltonian has been diagonalized properly

    :param eigenvalues:
    :param A: Matrix describing the network of oscillators
    :param K: Eigenvectors of A in a matrix form
    :return: True if diagonalized properly
    """
    a = np.diag(eigenvalues)
    return np.allclose(a, K.transpose() @ A @ K)


# Definition of the Wigner function for one oscillator
def wigner_function(cov: np.ndarray, vector: np.ndarray, exp_vector: np.ndarray) -> np.ndarray:
    """
    Calculates the value of the Wigner function in a given point.

    :param cov: Covariance matrix of the state
    :param vector: Where the Wigner function will be evaluated
    :param state_vector: Vector of the operators expectation values
    :return: wigner: value of the WIgner function
    """

    expectations = (vector - exp_vector)
    wigner = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov))) *\
        np.exp(-0.5 * expectations @ np.linalg.inv(cov) @ expectations)

    return wigner


# Vectorizing the function for convenience
np.vectorize(wigner_function)


def wigner_operations(grid: np.ndarray, cov_X: np.ndarray, storage: list, state_vector: np.ndarray) -> None:
    """
    Calculates all the values of the Wigner function in a given grid at a single point in time.

    :param grid: A square grid with a given resolution where the Wigner function will be evaluated
    :param cov_X: Covariance matrix of the state
    :param storage: Where the wigner functions values will be stored
    :param state_vector: Vector of the expectation values of the oscillator
    """

    # Where the current moments values will be stored in
    current_wigner_values = []

    # Iterates through the grid that the Wigner function will be calculated in row by row
    for row in grid:
        wig_row = []

        # Calculates the value of the Wigner function for every x and y coordinate pair in this row
        for i in range(RESOLUTION):
            wigner_value = wigner_function(cov_X, row[i], state_vector)
            wig_row.append(wigner_value)

        # The list containing all the Wigner function values for the specific row is added to a list
        # that contains every rows wigner function values
        current_wigner_values.append(np.array(wig_row))

    # The list containing all calculated Wigner function values for this instance in time is added to storage
    storage.append(np.array(current_wigner_values))


def wigner_visualization(W: np.ndarray, ax_range: np.ndarray) -> None:
    """Visualizes the distribution of the Wigner function in a 2D contour plot"""
    fig = plt.figure()
    ax = plt.axes(xlabel="Q", ylabel="P", aspect='equal')

    plt.contourf(ax_range, ax_range, W, 100)
    plt.colorbar()

    fig.suptitle("Wigner function")

    plt.show()


# Draws the frames for animating the Wigner function
def draw_frame(frame: int, W: np.ndarray, ax_range: np.ndarray) -> plt.Figure:
    """Draws a single frame for the animation of the Wigner function"""
    cont = plt.contourf(ax_range, ax_range, W[frame])
    return cont


def wigner_animation(W: np.ndarray, ax_range: np.ndarray, fig: plt.Figure, save=False) -> None:
    """Animates the Wigner function and saves the animation if save=True"""
    anim = ani.FuncAnimation(fig, draw_frame, frames=len(W), fargs=(W, ax_range), interval=200)

    if save:
        writer = PillowWriter(fps=30)
        anim.save("HeiluriSmooth.gif", writer=writer)

    plt.show()


# Creates still images depending on whether one or more oscillators are simulated
def visualisation(oscillators_var: dict, oscillators_exp: dict, oscillators_sq: dict, oscillators_phi: dict,
                  oscillators_exci: dict, oscillators_therm: dict, wigners: list, ax_range: np.ndarray) -> None:
    """
    Plots the variances and expectation values. Also makes matrix plots of the squeezing parameters and angles,
    excitations and thermal expectation values.

    :param oscillators_var: Dictionary of every oscillator's variances are saved
    :param oscillators_exp: Dictionary of every oscillator's operators expectation values are saved
    :param oscillators_sq: Dictionary of every oscillator's squeezing parameters are saved
    :param oscillators_phi: Dictionary of every oscillator's squeezing angles are saved
    :param oscillators_exci: Dictionary of every oscillator's excitations are saved
    :param oscillators_therm: Dictionary of every oscillator's thermal expectation values are saved
    :param wigners: List that contains all calculated Wigner functions values at different moments in time
    :param ax_range: Ranges for the plot axis
    """
    # Plotting if multiple oscillators, also animating wigner function if only one oscillator

    # If form one to ten oscillators are simulated, plots the variance of Q and P with respect to time
    # for every oscillator
    if N <= 10:
        for key, value in oscillators_var.items():
            plot_graphs(value["Q"], value["P"], TIMES, f"Variances for oscillator {key}")

        for key, value in oscillators_exp.items():
            if int(key) % 20 == 0:
                plot_graphs(value["Q"], value["P"], TIMES, f"Expectation values for oscillator {key}")

    # Separate contour plots
    plot_matrix(oscillators_sq, 'Squeezing parameter r', 'r')
    plot_matrix(oscillators_phi, 'Squeezing angle φ', 'φ')
    plot_matrix(oscillators_exci, 'Expectation value of excitations', 'Excitations')
    plot_matrix(oscillators_therm, 'Thermal excitations', 'Excitations')

    # If only one oscillator is simulated, draws a contour plot of the initial state and
    # also shows an animation of the time evolution of said state
    if N == 1:
        wigner_non = np.array(wigners)

        # Plots the WIgner function at t=0
        wigner_visualization(wigner_non[0], ax_range)

        fig = plt.figure()
        ax = plt.axes(xlim=(MIN_X_AND_Y, MAX_X_AND_Y),
                      ylim=(MIN_X_AND_Y, MAX_X_AND_Y), xlabel='Q', ylabel='P', aspect='equal')

        # Shows an animation of the Wigner functions time evolution
        wigner_animation(wigner_non, ax_range, fig, SAVE)


def plot_graphs(Q_values: list, P_values: list, times: np.ndarray, label: str) -> None:
    """
    Plots values with respect to time.

    :param Q_values: Expectation values or variances for Q
    :param P_values: Expectation values or variances for Q
    :param times: Values of time when the values were calculated
    :param label: What the plot will be named
    """

    # Creates three plots in one picture
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    # Plotting the values in the noninteracting base
    ax1.plot(times, [i[0] for i in Q_values], label="Q")
    ax1.plot(times, [i[0] for i in P_values], label="P")
    ax1.set_title("Noninteracting S, " + label)
    ax1.legend()

    # Plotting the values in the interacting base
    ax2.plot(times, [i[1] for i in Q_values], label="Q")
    ax2.plot(times, [i[1] for i in P_values], label="P")
    ax2.set_title("interacting S, " + label)
    ax2.legend()

    # Plotting the values in the old base
    ax3.plot(times, [i[2] for i in Q_values], label="Q")
    ax3.plot(times, [i[2] for i in P_values], label="P")
    ax3.set_title("Old S, " + label)
    ax3.legend()

    fig.tight_layout()

    plt.show()


def plot_single_parameter(parameter: np.ndarray, times: np.ndarray, label: str) -> None:
    """Plots the evolution of a single parameter (r, φ, ⟨n⟩ or n_th) for a single oscillator"""
    fig = plt.figure()
    ax = plt.axes(xlabel="t", ylabel="r", xlim=(0, times[-1]))

    ax.plot(times, parameter, label="r")
    ax.set_title(f"Squeeze for osc {label}")
    ax.legend()
    plt.show()


def plot_matrix(data: dict, label: str, color_label: str) -> None:
    """Plots a contour plot of a single matrix variable for a single oscillator"""
    matrix = np.array(list(data.values()))

    fig, ax = plt.subplots()
    ax.set_xlabel("Time [t]")
    ax.set_ylabel("Oscillator")
    # Possible cmaps: ocean, RdBu, gnuplot, inferno, cm.
    c = ax.pcolormesh(matrix, cmap=cm.blue_yellow, vmin=0, vmax=np.max(matrix))
    ax.set_title(label)

    fig.colorbar(c, ax=ax, label=color_label)

    # Setting x ticks so that there are five evenly spaced ticks with the correct time
    x = [DURATION * i / (TIME_STEP * 4) for i in range(5)]
    labels = [DURATION * i / 4 for i in range(5)]
    plt.xticks(x, labels)

    plt.show()


def save_parameters(t: float, cov: np.ndarray, sq_dict: dict, phi_dict: dict, exci_dict: dict, therm_dict: dict,
                    frequencies: np.ndarray, first_moments: np.ndarray) -> None:
    """
    Saves all of the the squeezing parameters and -angles, excitations and thermal expectation values into dictionaries.

    :param t: Moment in tiime
    :param cov: Covariance matrix of the whole network
    :param sq_dict: Where the squeezing parameters will be saved
    :param phi_dict: Where the squeezing angles will be saved
    :param exci_dict: Where the operator expectation values will be saved
    :param therm_dict: Where the thermal expectation values will be saved
    :param frequencies: Oscillators frequencies that are used in calculations
    :param first_moments: Vector of the operators expectation values
    :return: None
    """

    # Iterates through every oscillator
    for i in range(N):
        freq = frequencies[i]

        # Picks up the covariance matrix describing the particular oscillator
        block_matrix = np.array([[cov[i][i], cov[i][N + i]], [cov[N + i][i], cov[i + N][i + N]]])

        # Calculates the number of thermal excitations
        n = thermal_excitations(block_matrix)

        # Calculates the average number of photons in each state
        first_moments_i = np.array([first_moments[i], first_moments[i + N]])
        excitations = get_excitations(block_matrix, first_moments_i, freq)

        # Due to systematic calculation errors, the argument in cosh() is sometimes
        # insignificantly under 1, so let's round this up to 1
        argument = max((block_matrix[0][0] * freq + block_matrix[1][1] / freq) / (1 + 2 * n), 1)

        # If first calculated value, create a list, else just add to the existing list
        if t == 0:
            # Calculated value of the squeezing parameter r
            r = 0.5 * np.arccosh(argument)

            # Calculate squeezing angle with r and off
            if r != 0:
                phi_dict[f"{i}"] = [block_matrix[0][1] / (np.sinh(2 * r) * (1 + 2 * n))]
            else:
                phi_dict[f"{i}"] = [0]

            therm_dict[f"{i}"] = [n]
            sq_dict[f"{i}"] = [r]
            exci_dict[f"{i}"] = [excitations]

        else:
            r = 0.5 * np.arccosh(argument)

            if r != 0:
                phi_dict[f"{i}"].append(block_matrix[0][1] / (np.sinh(2 * r) * (1 + 2 * n)))
            else:
                phi_dict[f"{i}"].append(0)

            therm_dict[f"{i}"].append(n)
            sq_dict[f"{i}"].append(r)
            exci_dict[f"{i}"].append(excitations)


def get_excitations(cov: np.ndarray, first_moments: np.ndarray, freq: float) -> float:
    """
    Calculates the amount of excitations/photons in the state

    :param cov: covariance matrix of the state
    :param first_moments: Vector containing expectation values of the operator
    :param freq: frequency of the oscillator
    :return: total number of excitations/photons
    """
    excitations = 0.5 * (cov[0][0] * freq + cov[1][1] / freq + sum(first_moments ** 2) - 1)
    return excitations


def main():
    """Main program"""
    # Starting the runtime -timer
    start = timeit.default_timer()

    # The network
    G = network_initialize(True)
    K, eigenfrequencies = matrix_K(G, FREQUENCIES)

    # Range for where Wigner function is calculated
    ax_range = np.linspace(MIN_X_AND_Y, MAX_X_AND_Y, RESOLUTION)

    # Creates a grid of points for where the wigner function is evaluated in
    grid = np.array([[(j, i) for j in ax_range] for i in ax_range])

    # Dictionaries where every oscillators' expectation values, variances, squeezing parameters and -angles,
    # excitations and thermal excitations will be saved to
    oscillators_exp = {}
    oscillators_var = {}
    oscillators_sq = {}
    oscillators_phi = {}
    oscillators_exci = {}
    oscillators_therm = {}

    # Storage for the values of the Wigner function at each instance in time
    wigner_values = []

    # Creating the initial covariance matrices. The interacting base uses the noninteracting covariance
    # matrix so only two calculations are needed
    cov_X_initial_non = initial_cov_matrix(eigenfrequencies, SQUEEZING_PARAMETERS, SQUEEZING_ANGLES)
    cov_X_initial_old = initial_cov_matrix(FREQUENCIES, SQUEEZING_PARAMETERS, SQUEEZING_ANGLES)

    # Iterates the initial state in three different bases
    for t in TIMES:
        # The corresponding symplectic matrix S in the non-interacting base for the value of t.
        # New covariance matrix and vector of expectation values at moment t are also calculated
        S_non = symplectic(t, K, eigenfrequencies, "non_interacting")
        variances_non, new_cov_X_non = new_cov_calculation(S_non, cov_X_initial_non)
        expectation_values_non = S_non @ INITIAL_STATE_VECTOR

        # S, cov(X) and ⟨x⟩ in the interacting base
        S_int = symplectic(t, K, eigenfrequencies, "interacting")
        variances_int, new_cov_X_int = new_cov_calculation(S_int, cov_X_initial_non)
        expectation_values_int = S_int @ INITIAL_STATE_VECTOR

        # S, cov(X) and ⟨x⟩ in the old base
        S_old = symplectic(t, K, eigenfrequencies, "old")
        variances_old, new_cov_X_old = new_cov_calculation(S_old, cov_X_initial_old)
        expectation_values_old = S_old @ INITIAL_STATE_VECTOR

        # Saves the variances, expectation values, squeezing parameters and -angles,
        # excitations and thermal excitations into dictionaries
        save_into_dict(t, oscillators_var, variances_non, variances_int, variances_old)
        save_into_dict(t, oscillators_exp, expectation_values_non, expectation_values_int, expectation_values_old)
        save_parameters(t, new_cov_X_old, oscillators_sq, oscillators_phi,
                        oscillators_exci, oscillators_therm, FREQUENCIES, expectation_values_old)

        # A check is done that all the S matrices are still symplectic
        if not sanity_check_1(S_non, S_int, S_old):
            print("The matrix S is not symplectic anymore.")
            break

        # If only one oscillator is simulated, the corresponding wigner functions values are calculated
        if N == 1:
            state_vector = S_non @ INITIAL_STATE_VECTOR
            wigner_operations(grid, new_cov_X_non, wigner_values, state_vector)

    visualisation(oscillators_var, oscillators_exp, oscillators_sq, oscillators_phi,
                  oscillators_exci, oscillators_therm, wigner_values, ax_range)

    stop = timeit.default_timer()
    print(f"\nRuntime: {(stop - start):.4f} s.")


if __name__ == '__main__':
    main()
