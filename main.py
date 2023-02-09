import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import timeit
import networkx as nx
from random import randint
from json import load
from matplotlib.animation import PillowWriter
#import matplotlib;matplotlib.use("TkAgg")

# Reading the config file and setting the constants values
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

# The frequencies of the oscillators
FREQUENCIES = np.ones(N)
# frequencies = np.array([2.0])

####
#SQUEEZING_PARAMETERS = np.zeros(200)
#SQUEEZING_PARAMETERS[0] = 0.2
#SQUEEZING_ANGLES = np.zeros(200)
#INITIAL_STATE_VECTOR = np.zeros(400)
####

#print("Squeezing parameters")
#print(SQUEEZING_PARAMETERS)
#print("\nSqueezing angles")
#print(SQUEEZING_ANGLES)

#print("\nvector of first moments")
#print(INITIAL_STATE_VECTOR)

# Defining a NxN zero matrix to help construct the other matrices
zero_block = np.zeros((N, N))

# Defining the J matrix
J = np.block([[zero_block, np.identity(N)],
              [-1 * np.identity(N), zero_block]])


def get_excitations(cov: np.ndarray, first_moments: np.ndarray) -> float:
    """
    Calculates the amount of excitations/photons in the system

    :param cov: covariance matrix
    :param first_moments: Vector containing expectation values of the operators
    :return: total number of excitations/photons
    """
    excitations = 0.5 * (np.trace(cov) + sum(first_moments ** 2) - 1)
    return excitations


def thermal_excitations(cov: np.ndarray) -> float:
    """
    Calculates the amount of thermal excitations in the system

    :param cov: covariance matrix of the system
    :return: amount of thermal excitations
    """
    th_excitations = np.sqrt(np.linalg.det(cov)) - 0.5
    return th_excitations


# Calculates the expectation value of the operators Q_i squared
def expectation_value_Q2(frequency: np.ndarray) -> np.ndarray:
    """
        Args: frequency: List of the oscillators frequencies

        Return: array of
    """

    if T == 0:
        values = 0.5 / frequency

    else:
        values = (1 / (np.exp(frequency / T) - 1) + 0.5) / frequency

    return values


# Calculates the expectation value of the operators P_i squared
def expectation_value_P2(frequency: np.ndarray) -> np.ndarray:
    if T == 0:
        value = 0.5 * frequency

    else:
        value = (1 / (np.exp(frequency / T) - 1) + 0.5) * frequency

    return value


def initial_cov_matrix(freqs: np.ndarray, Q2_expectation: np.ndarray, P2_expectation: np.ndarray,
                       squeezing_parameters: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """
    Calculates the initial covariance matrix of the whole system

    :param freqs: frequencies
    :param Q2_expectation: Expectation values of the squares of the Q operators
    :param P2_expectation: Expectation values of the squares of the P operators
    :param sq_r: The squeezing parameters
    :param sq_phi: The squeezing angles
    :return: covariance matrix
    """

    # Creates a template for the covariance matrix
    cov_X_initial = np.zeros((2 * N, 2 * N))

    # Oikein ???

    """
    Q2_multipliers_func = np.vectorize(lambda r, p, f: f**2 * np.sin(p)**2 *
                                       np.sinh(r)**2 + (np.cosh(r) + np.cos(p) * np.sinh(r))**2)

    Q2_multipliers = Q2_multipliers_func(sq_r, sq_phi, freqs)

    P2_multipliers_func = np.vectorize(lambda r, p, f: (1 / f**2) * np.sin(p)**2 *
                                       np.sinh(r)**2 + (np.cosh(r) - np.cos(p) * np.sinh(r))**2)
    P2_multipliers = P2_multipliers_func(sq_r, sq_phi, freqs)

    cov_X_initial += np.diag(np.concatenate((Q2_expectation * Q2_multipliers, P2_expectation * P2_multipliers)))
    """

    print(cov_X_initial)

    # Checking the indices of vacuum states
    vacuum_indices = np.where(SQUEEZING_PARAMETERS == -1)[0]
    print(f"\nVacuum states {vacuum_indices}")

    # Checks which oscillators have squeezed states and which don't have
    squeezed_indices = np.where(SQUEEZING_PARAMETERS > 0)[0]
    print(f"Indices of squeezed oscillators: {squeezed_indices}")

    # Indices of thermal states
    thermal_indices = np.where(SQUEEZING_PARAMETERS == 0)[0]
    print(f"Thermal_indices: {thermal_indices}")

    # Calculating the initial variances for every oscillator:

    # Calculating the thermal expectation value
    # If vacuum state, use limit for thermal expectation at T=0
    thermal_expectations = []

    # Iterates through each state
    for i in range(N):
        print(phis)
        phi = phis[i]
        r = squeezing_parameters[i]
        freq = freqs[i]

        print(f"Osc {i}: phi={phi}, r={r}, freq={freq}")

        # Defining the symplectic squeezing matrix
        symp_squeezing_matrix = np.zeros((2, 2))
        symp_squeezing_matrix[0][0] = np.cosh(r) + np.sinh(r) * np.cos(phi)
        symp_squeezing_matrix[1][1] = np.cosh(r) - np.sinh(r) * np.cos(phi)
        symp_squeezing_matrix[1][0] = freq * np.sinh(r) * np.sin(phi)
        symp_squeezing_matrix[0][1] = (1 / freq) * np.sinh(r) * np.sin(phi)
        print(symp_squeezing_matrix)

        # If simulating a vacuum state or a squeezed vacuum state use hand calculated limit for the general
        # case of thermal states. Else use the general thermal state version with temperature T
        if T == 0 or i in vacuum_indices:
            thermal_expectation_term = 0.5
        else:
            thermal_expectation_term = 1 / (np.exp(freq / T) - 1) + 0.5

        print("Thermal expectation value:", thermal_expectation_term)
        cov_osc_i = np.diag(np.array([thermal_expectation_term / freq, thermal_expectation_term * freq]))

        # cov_X_initial[i][i + N] = thermal_expectation * \
                                # np.sinh(2 * SQUEEZING_PARAMETERS[i]) * np.sin(SQUEEZING_ANGLES[i])

        #cov_X_initial[i][i + N] = thermal_expectation * np.sin(phi) * np.sinh(r) *\
                                  #((freqs[i] + 1 / freqs[i]) * np.cosh(r) -
                                   #(freqs[i] - 1 / freqs[i]) * np.cos(phi) * np.sinh(r))

        #cov_X_initial[i + N][i] = cov_X_initial[i][i + N]

        # Perform squeezing if the state is squeezed
        if i in squeezed_indices:
            cov_osc_i = symp_squeezing_matrix.transpose() @ cov_osc_i @ symp_squeezing_matrix

        cov_X_initial[i][i] = cov_osc_i[0][0]
        cov_X_initial[i + N][i + N] = cov_osc_i[1][1]
        cov_X_initial[i + N][i] = cov_osc_i[1][0]
        cov_X_initial[i][i + N] = cov_osc_i[0][1]

        print("Initial cov")
        print(cov_osc_i)

    print("Combined initial covariance matrix:")
    print(cov_X_initial)
    print(f"Frequencies: {freqs}")

    return cov_X_initial


def new_cov_calculation(S: np.ndarray, cov_X: np.ndarray, type: str) -> tuple:
    new_cov_X = S @ cov_X @ S.transpose()
    diagonal = np.diagonal(new_cov_X)
    return diagonal, new_cov_X


def covariance_matrix_save(t: float, covariance: list, covariance_matrix: np.ndarray) -> None:
    covariance.append((t, covariance_matrix))


def save_into_dict(t: float, storage: dict, non: np.ndarray,
                   int: np.ndarray, old: np.ndarray) -> None:
    """
    Saves either the variances or expectation values into a corresponding dict

    :param t: Moment in time
    :param storage: The dict into where the data is saved
    :param non: The value in non interacting base
    :param int: The value in interacting base
    :param old: The value in the old base
    :return: None
    """
    # Iterates through every oscillator in the system
    for i in range(N):
        # Need to initialize the lists and dicts if t=0
        if t == 0:
            storage[f"{i}"] = {'Q': [(non[i], int[i], old[i])],
                                   'P': [(non[i + N], int[i + N], old[i + N])]}

        # If dicts and lists are already there, just append the new values to the lists
        else:
            storage[f"{i}"]['Q'].append((non[i], int[i], old[i]))
            storage[f"{i}"]['P'].append((non[i + N], int[i + N], old[i + N]))


# Applies the cos(Ωt) operation on each frequency
def cosine(frequency: np.ndarray, t: float) -> np.ndarray:
    return np.cos(frequency * t)


# Vectorizes the function for convenience
cosine_vector = np.vectorize(cosine)


# Applies the sin(Ωt) operation on each frequency
def sine(frequency: np.ndarray, t: float) -> np.ndarray:
    return np.sin(frequency * t)


sine_vector = np.vectorize(sine)


# Creates the network
def network_initialize(plot=False):
    # The graph containing N quantum mechanical harmonic oscillators
    if N == 1:
        G = nx.newman_watts_strogatz_graph(N, 1, 0.5)
    else:
        G = nx.newman_watts_strogatz_graph(N, 2, 0.0)

    # Adding random weights
    for u, v in G.edges():
        # G.edges[u, v]['weight'] = randint(1, 5)
        # print(u, v)
        G.edges[u, v]['weight'] = 1

    ######
    G.edges[0, 9]['weight'] = 0
    ######

    # Draws the network
    if plot and N > 1:
        nx.draw_networkx_labels(G, pos=nx.circular_layout(G))
        nx.draw_circular(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=labels)

    return G


def matrix_K(G):
    # Construction of matrix A:

    # Adjacency matrix
    V = nx.adjacency_matrix(G).toarray()

    # Degree matrix
    D = np.diag([val for (node, val) in sorted(G.degree(), key=lambda pair: pair[0])])
    # print(V)
    # print(D)

    # Laplacian matrix
    L = D - V
    print("L")
    print(L)

    # Finally, matrix A
    print(f"Frequencies for A: {FREQUENCIES}")
    A = np.diag(FREQUENCIES ** 2) * 0.5 + 0.5 * L
    print("A matriisi:")
    print(A)
    print()

    eigenvalues, K = np.linalg.eigh(A)
    print("eigenvalues", eigenvalues)
    eigenfrequencies = np.sqrt(eigenvalues * 2)

    print(sanity_check_2(eigenvalues, K, A))

    if not sanity_check_2(eigenvalues, K, A):
        raise Exception("Diagonalization of matrix A failed")

    return K, eigenfrequencies


# Creates the symplectic matrix for a specific moment in time
def symplectic(t: float, K: np.ndarray, freqs: np.ndarray, base: str) -> np.ndarray:

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

    # If the given base is not possible
    else:
        raise ValueError("Function symplectic() argument 'kind' invalid")

    return S


# Checks whether the symplectic matrix is really symplectic
def sanity_check_1(S1: np.ndarray, S2: np.ndarray, S3: np.ndarray) -> bool:
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


# Checks if the Hamiltonian has been diagonalized properly
def sanity_check_2(eigenvalues, K: np.ndarray, A: np.ndarray) -> bool:
    a = np.diag(eigenvalues)
    return np.allclose(a, K.transpose() @ A @ K)


# Definition of the Wigner function for one oscillator
def wigner_function(cov, vector, state_vector) -> np.ndarray:
    oe = (vector - state_vector)
    wigner = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov))) *\
        np.exp(-0.5 * oe @ np.linalg.inv(cov) @ oe)

    return wigner


np.vectorize(wigner_function)


def wigner_operations(grid: np.ndarray, cov_X: np.ndarray, storage: list, state_vector: np.ndarray):
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


# Visualizes the distribution of the Wigner function in a 2D contour plot
def wigner_visualization(W, ax_range):
    fig = plt.figure()
    ax = plt.axes(xlabel="Q", ylabel="P", aspect='equal')

    plt.contourf(ax_range, ax_range, W, 100)
    plt.colorbar()

    fig.suptitle("Wigner function")

    plt.show()


# Draws the frames for animating the Wigner function
def draw_frame(frame, W, ax_range) -> plt.Figure:
    cont = plt.contourf(ax_range, ax_range, W[frame])
    return cont


# Runs animation
def wigner_animation(W, ax_range, fig, save=False):
    anim = ani.FuncAnimation(fig, draw_frame, frames=len(W), fargs=(W, ax_range), interval=200)

    if save:
        writer = PillowWriter(fps=30)
        anim.save("HeiluriSmooth.gif", writer=writer)

    plt.show()


# Creates still images depending on whether one or more oscillators are simulated
def visualisation(oscillators_var: dict, oscillators_exp: dict, wigners: list, ax_range: np.ndarray):
    # Plotting if multiple oscillators, also animating wigner function if only one oscillator

    # If more than one oscillator simulated, plots the variance of Q and P with respect to time
    # for every oscillator

    for key, value in oscillators_var.items():
        plot_graphs(value["Q"], value["P"], TIMES, f"Variances for oscillator {key}")

    for key, value in oscillators_exp.items():
        plot_graphs(value["Q"], value["P"], TIMES, f"Expectation values for oscillator {key}")

        # Testing only
        # plot_expectations(value["Exp"]['Q'], value["Exp"]['P'], TIMES, f"Expectation values for oscillator {key}")

    # If only one oscillator is simulated, draws a contour plot of the initial state and
    # also shows an animation of the time evolution of said state
    if N == 1:
        wigner_non = np.array(wigners)
        wigner_visualization(wigner_non[0], ax_range)

        fig = plt.figure()
        ax = plt.axes(xlim=(MIN_X_AND_Y, MAX_X_AND_Y),
                      ylim=(MIN_X_AND_Y, MAX_X_AND_Y), xlabel='Q', ylabel='P', aspect='equal')

        wigner_animation(wigner_non, ax_range, fig, SAVE)


# Plots the expectation values with respect to time
def plot_graphs(Q_values: list, P_values: list, times: np.ndarray, label: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(times, [i[0] for i in Q_values], label="Q")
    ax1.plot(times, [i[0] for i in P_values], label="P")
    ax1.set_title("Noninteracting S, " + label)
    ax1.legend()

    ax2.plot(times, [i[1] for i in Q_values], label="Q")
    ax2.plot(times, [i[1] for i in P_values], label="P")
    ax2.set_title("interacting S, " + label)
    ax2.legend()

    ax3.plot(times, [i[2] for i in Q_values], label="Q")
    ax3.plot(times, [i[2] for i in P_values], label="P")
    ax3.set_title("Old S, " + label)
    ax3.legend()

    fig.tight_layout()

    plt.show()


def plot_expectations(Q_values: list, P_values: list, times: np.ndarray, label: str):
    fig = plt.figure()
    ax = plt.axes(xlabel="Q", ylabel="P", aspect='equal')


    ax.plot(times, [i[0] for i in Q_values], label="Q")
    ax.plot(times, [i[0] for i in P_values], label="P")
    ax.set_title("Noninteracting S")
    ax.legend()


def photon_expectation_value(cov: np.ndarray, first_moments: np.ndarray) -> list:
    """

    :param cov:np.ndarray Covariance matrix of the state
    :param first_moments:  Vector of first moments
    :return:
    """
    photon_exp = 0.5 * (np.trace(cov) + sum(first_moments ** 2) - 1)

    return photon_exp


def get_squeezing_parameters(cov: np.ndarray) -> float:
    """

    :param cov: Covariance matrix of the system
    :return: float: The squeezing parameter of the states
    """

    squeeze_r = []

    for i in range(N):
        block_matrix = np.array([[cov[i][i], cov[i][N + i]], [cov[N + i][i], cov[i + N][i + N]]])
        # print("Squeeze")
        # print(block_matrix)

        # Joku muunnos tarvitaan vielä...
        n = np.sqrt(np.linalg.det(block_matrix)) - 0.5
        # Varmaan tähänkin

        # Due to systematic calculation errors, the argument in cosh() is sometimes
        # like 0.99999999999, so let's round this up to 1
        argument = max(np.trace(block_matrix) / (1 + 2 * n), 1)

        squeeze_r.append((i, 0.5 * np.arccosh(argument)))

    return squeeze_r


def thermal_excitation():
    pass


def main():
    # Starting the runtime -timer
    start = timeit.default_timer()

    # The network
    G = network_initialize(True)
    K, eigenfrequencies = matrix_K(G)
    print("K matriisi")
    print(K)
    # print("Matrix K:")
    # print(K)

    # Range for where Wigner function is calculated
    ax_range = np.linspace(MIN_X_AND_Y, MAX_X_AND_Y, RESOLUTION)

    # Creates a grid for the evaluation of the wigner function
    grid = []
    for i in ax_range:
        row = []
        for j in ax_range:
            row.append((j, i))
        grid.append(np.array(row))
    grid = np.array(grid)

    # A dictionary into which every oscillators' variance values will be saved to
    oscillators_var = {}

    # A dictionary into which every oscillators' expectation values will be saved to
    oscillators_exp = {}

    # A list where every covariance matrix will be stored
    covariance_matrices = []

    # Storage for the values of the Wigner function at each instance in time
    wigner_values = []

    # Storage for the calculated squeezing parameters
    squeezed_indices = np.nonzero(SQUEEZING_PARAMETERS)[0]
    squeezes = []

    # Vectorizes the function that calculates the expectation values for Q^2 and applies it to get the
    # expectation values in an array form
    E_Q2_vector = np.vectorize(expectation_value_Q2)
    Q2_expectation = E_Q2_vector(eigenfrequencies)
    q2_expectation = E_Q2_vector(FREQUENCIES)

    # Does the same vectorization and calculations for P^2 as before for Q^2
    E_P2_vector = np.vectorize(expectation_value_P2)
    P2_expectation = E_P2_vector(eigenfrequencies)
    p2_expectation = E_Q2_vector(FREQUENCIES)

    ##### Random testing a trying to reformat

    #print(Q2_expectation)
    #print(P2_expectation)
    print(eigenfrequencies)
    cov_X_initial_non = initial_cov_matrix(eigenfrequencies, Q2_expectation, P2_expectation,
                                       SQUEEZING_PARAMETERS, SQUEEZING_ANGLES)
    cov_X_initial_old = initial_cov_matrix(FREQUENCIES, q2_expectation, p2_expectation,
                                           SQUEEZING_PARAMETERS, SQUEEZING_ANGLES)

    #####

    """
    print(cov_X_initial)
    print(f"Determinant: {np.linalg.det(cov_X_initial)}")
    print(f"Q2: {Q2_expectation}")
    print(f"P2: {P2_expectation}")
    print(f"Purity: {1/(2 * np.sqrt(np.linalg.det(cov_X_initial)))}")
    """

    print("\n\n\n\n\n")

    # Iterates the initial state in three different bases
    for t in TIMES:
        # Diagonals should really be the expectation values and also covariance_save() should be rewritten
        # and renamed

        # The corresponding symplectic matrix S in the non-interacting base for the value of t
        S_non = symplectic(t, K, eigenfrequencies, "non_interacting")
        variances_non, new_cov_X_non = new_cov_calculation(S_non, cov_X_initial_non, "non")
        expectation_values_non = S_non @ INITIAL_STATE_VECTOR

        #print("initial state vector")
        #print(initial_state_vector)
        #print("new state vector")
        #print(expectation_values_non)

        # S in the interacting base
        S_int = symplectic(t, K, eigenfrequencies, "interacting")
        variances_int, new_cov_X_int = new_cov_calculation(S_int, cov_X_initial_non, "int")
        expectation_values_int = S_int @ INITIAL_STATE_VECTOR
        #print(new_cov_X_int)

        # S in the old base
        S_old = symplectic(t, K, eigenfrequencies, "old")
        variances_old, new_cov_X_old = new_cov_calculation(S_old, cov_X_initial_old, "old")
        expectation_values_old = S_old @ INITIAL_STATE_VECTOR

        print("Symplectic S_non, t =", t)
        print(S_non)
        print()

        print("Symplectic S_int, t =", t)
        print(S_int)
        print()

        print("Symplectic S_old, t =", t)
        print(S_old)
        print()

        #print(S_old)

        #print()
        #print("new_cov_X_old")
        #print(new_cov_X_old)

        #if t == 1:
            #print("Covs at t==0")
            #print(new_cov_X_non)
            #print(new_cov_X_int)
            #print(new_cov_X_old)

        # Saves the three current covariance matrices
        save_into_dict(t, oscillators_var, variances_non, variances_int, variances_old)
        save_into_dict(t, oscillators_exp, expectation_values_non, expectation_values_int, expectation_values_old)

        # If at the start there were squeezed states save all the squeezing parameters of every oscillator
        squeezes.append(get_squeezing_parameters(new_cov_X_old))

        # A check is done that all the S matrices are still symplectic
        if not sanity_check_1(S_non, S_int, S_old):
            print("The matrix S is not symplectic anymore.")
            break

        # If only one oscillator is simulated, the corresponding wigner function is calculated
        if N == 1:
            state_vector = S_non @ INITIAL_STATE_VECTOR
            wigner_operations(grid, new_cov_X_non, wigner_values, state_vector)

    # print("\nexp old:")
    # print(expectation_values_old)


    # if there are squeezed states print all squeezing parameters
    # if len(squeezed_indices) and N > 1:
    print("Squeezes")
    #print(squeezes)
    for i in squeezes:
        print(i)

    stop = timeit.default_timer()
    print(f"\nRuntime: {(stop - start):.4f} s.")

    visualisation(oscillators_var, oscillators_exp, wigner_values, ax_range)
    # print(oscillators_exp)


if __name__ == '__main__':
    main()
