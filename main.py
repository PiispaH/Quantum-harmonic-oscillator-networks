import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
#import matplotlib;matplotlib.use("TkAgg")
import timeit
import networkx as nx
from random import randint
#from matplotlib.animation import PillowWriter

# The amount of oscillators and system temperature
N = 3
T = 1

# Coordinates for the initial state
r = 0.1
phi = 0.1

# Constants related to simulation time and plotting
TIMES = [i/5 for i in range(20)]
MIN = -5
MAX = 5
RESOLUTION = 51

# Defining a NxN zero matrix to help construct the other matrixes
zero = np.zeros((N, N))

# Defining the J matrix
J = np.block([[zero, np.identity(N)],
              [-1 * np.identity(N), zero]])

# The frequencies of the oscillators
frequencies = np.ones(N)
"""frequencies = []
for i in range(N):
    frequencies.append(0.1 + i * 0.1)
frequencies = np.array(frequencies)"""
#random_state = [randrange(1, 5) for i in range(2 * N)]
#initial_state_vector = np.array(random_state)


# Calculates the expectation value of the operators Q_i squared
def expectation_value_Q2(frequency):
    if T == 0:
        value = 0.5 / frequency

    else:
        value = (1 / (np.exp(frequency / T) - 1) + 0.5) / frequency

    return value


# Vectorizes the function and applies it to get the expectation values for Q^2
E_Q2_vector = np.vectorize(expectation_value_Q2)
Q2_expectation = E_Q2_vector(frequencies)


# Calculates the expectation value of the operators P_i squared
def expectation_value_P2(frequency):
    if T == 0:
        value = 0.5 * frequency

    else:
        value = (1 / (np.exp(frequency / T) - 1) + 0.5) * frequency

    return value


# Vectorizes the function and applies it to get the expectation values for P^2
E_P2_vector = np.vectorize(expectation_value_P2)
P2_expectation = E_P2_vector(frequencies)


def initial_cov_matrix(N: int, zero, frequencies):
    if N == 1:
        # Creating the parts presenting the squeezing and displacement operators in the covariance matrix
        Q2_multiplier = []
        P2_multiplier = []

        for i in range(N):
            Q2_multiplier.append(np.cosh(2 * r) + np.sinh(2 * r) * np.cos(phi))
            P2_multiplier.append(np.cosh(2 * r) - np.sinh(2 * r) * np.cos(phi))

        off_diagonal = - (1 / (np.exp(frequencies / T) - 1) + 0.5) * np.sinh(2 * r) * np.sin(phi)

        # Creating the covariance matrix describing the initial state
        cov_X_initial = np.block([[np.diag(Q2_expectation * Q2_multiplier), off_diagonal],
                                  [off_diagonal, np.diag(P2_expectation * P2_multiplier)]])

    else:
        cov_X_initial = np.block([[np.diag(Q2_expectation), zero],
                                  [zero, np.diag(P2_expectation)]])

    return cov_X_initial

#
# Constructing the symplectic matrix:
#

# Applies the cos(Ωt) operation on each frequency
def cosine(frequency, t):
    return np.cos(frequency * t)


cosine_vector = np.vectorize(cosine)


# Applies the sin(Ωt) operation on each frequency
def sine(frequency, t):
    return np.sin(frequency * t)


sine_vector = np.vectorize(sine)


# Creates the symplectic matrix for a specific moment in time
def symplectic(t, K, interacting=False):
    block_cos = np.diag(cosine_vector(frequencies, t))
    block_sin = np.diag(sine_vector(frequencies, t))

    freq_diag = np.diag(frequencies)
    inverse_freq_diag = np.linalg.inv(freq_diag)

    if interacting:
        S = np.block([[K @ block_cos, K @ (inverse_freq_diag * block_sin)],
                      [K @ (-freq_diag * block_sin), K @ block_cos]])
    else:
        S = np.block([[block_cos, inverse_freq_diag * block_sin],
                      [-freq_diag * block_sin, block_cos]])
    return S


def vector_multiplication(S, vector):
    return S @ vector


def matrix_A(graph, frequencies):
    laplacian = nx.laplacian_matrix(graph).toarray()
    freq = np.diag(frequencies) ** 2

    return freq * 0.5 + laplacian * 0.5


# Checks whether the symplectic matrix is really symplectic
def sanity_check_1(J, S):
    new_J_matrix = S @ J @ np.transpose(S)
    return np.allclose(J, new_J_matrix)


def sanity_check_2(eigenvalues, K, A):
    a = np.diag(eigenvalues)
    return np.allclose(a, K.transpose() @ A @ K)


# For animation
def draw(frame, Q, P):
    plt.clf()
    ax = plt.axes()

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    # Piirtää kuulat oikeille paikoille, ensimmäinen kuula pysyy aina paikallaan keskellä kuvaa
    plt.plot((0.5, Q[frame][0]), (0.5, P[frame][0]), 'ko', linestyle="-",
             ms=12, lw=2.5)
    plt.plot((Q[frame][0], Q[frame][1]), (P[frame][0], P[frame][1]), 'ko', linestyle="-",
             ms=12, lw=2.5)


# Plots the expectation values with respect to time
def plot_expectation(Q_values: list, P_values: list, times: list, label: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(times, [i[0] for i in Q_values], label="Q")
    ax1.plot(times, [i[0] for i in P_values], label="P")
    ax1.set_title("Noninteracting S")
    ax1.legend()

    ax2.plot(times, [i[1] for i in Q_values], label="Q")
    ax2.plot(times, [i[1] for i in P_values], label="P")
    ax2.set_title("Interacting S")
    ax2.legend()

    ax3.plot(times, [i[2] for i in Q_values], label="Q")
    ax3.plot(times, [i[2] for i in P_values], label="P")
    ax3.set_title("Old S")
    ax3.legend()

    fig.tight_layout()

    plt.show()


def wigner_visualization(W, ax_range):
    fig = plt.figure()

    plt.contourf(ax_range, ax_range, W, 100)
    plt.colorbar()

    fig.suptitle("Wigner function")

    plt.show()


def wigner_plot_ani(frame, W, ax_range):
    """fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5), xlabel='x', ylabel='y')
    plt.contour(ax_range, ax_range, W[frame])
    plt.colorbar()
    #fig.suptitle("Wigner function")"""

    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5), xlabel='x', ylabel='y')

    cont = plt.contourf(ax_range, ax_range, W[frame])  # first image on screen
    plt.colorbar()


def test_animation(frame, W, ax_range, cont):
    #print(cont.collections)
    #for c in cont.collections:
        #c.remove()

    cont = plt.contourf(ax_range, ax_range, W[frame])
    return cont


# For animation
def wigner_animation(W, ax_range, fig, cont):
    anim = ani.FuncAnimation(fig, test_animation, frames=len(W), fargs=(W, ax_range, cont))
    plt.show()
    #writer = PillowWriter(fps=30)
    #anim.save("Heiluri.gif", writer=writer)


def wigner_manual(cov, vector):
    oe = vector - np.array([0, 0])
    jees = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 * oe @ np.linalg.inv(cov) @ np.transpose(oe))

    return jees


np.vectorize(wigner_manual)


# Creates the network
def network(N: int, plot=False):
    # The graph containing N quantum mechanical harmonic oscillators
    if N == 1:
        G = nx.newman_watts_strogatz_graph(N, 1, 0.5)
    else:
        G = nx.newman_watts_strogatz_graph(N, 2, 0.5)

    # Adding random weights
    for u, v in G.edges():
        G.edges[u, v]['weight'] = randint(1, 10)

    # Construction of matrix A
    A = matrix_A(G, frequencies)

    # Draws the network
    if plot:
        nx.draw_circular(G)

    eigenvalues, K = np.linalg.eigh(A)
    print(f"Diagonalized properly: {sanity_check_2(eigenvalues, K, A)}")

    return G, A, eigenvalues, K


def wigner_operations(grid, RESOLUTION, cov_X, storage: list):
    temp_wig_storage = []
    for row in grid:
        wig_row = []
        for i in range(RESOLUTION):
            wiggy = wigner_manual(cov_X, row[i])
            wig_row.append(wiggy)
        temp_wig_storage.append(np.array(wig_row))
    storage.append(np.array(temp_wig_storage))


def new_cov_calculation(S, cov_X):
    new_cov_X = S @ cov_X @ S.transpose()
    diagonal = np.diagonal(new_cov_X)
    return diagonal, new_cov_X


def main():
    # Starting the runtime -timer
    start = timeit.default_timer()

    # The network
    G, A, eigenvalues, K = network(N, True)

    # Range for where Wigner function is calculated
    ax_range = np.linspace(MIN, MAX, RESOLUTION)

    # Creates a grid for the evaluation of the wigner function
    grid = []
    for i in ax_range:
        row = []
        for j in ax_range:
            row.append((j, i))
        grid.append(np.array(row))
    grid = np.array(grid)

    # A dict into which every oscillators' expectation values will be saved to
    oscillators = {}

    # Storage for the values of the Wigner function at each moment
    wigner_non = []

    # Initial covariance matrix of the network
    cov_X_initial = initial_cov_matrix(N, zero, frequencies)

    # Iterates the initial state in three different bases
    for t in TIMES:
        # The corresponding symplectic matrix S in the non-interacting base for the value of t
        S_noninteracting = symplectic(t, K)
        diagonal_non, new_cov_X = new_cov_calculation(S_noninteracting, cov_X_initial)
        if N == 1:
            wigner_operations(grid, RESOLUTION, new_cov_X, wigner_non)

        # S in the interacting base
        S_interacting = symplectic(t, K, interacting=True)
        diagonal_int, new_cov_X = new_cov_calculation(S_interacting, cov_X_initial)

        # S in the old base
        X = np.block([[K.transpose(), zero],
                      [zero, K.transpose()]])
        S_old = X.transpose() @ S_noninteracting @ X
        diagonal_old, new_cov_X = new_cov_calculation(S_old, cov_X_initial)

        # Goes through every pair of expectation values in the covariance matrix
        for i in range(N):
            if t == 0:
                oscillators[f"{i}"] = {'Q': [(diagonal_non[i], diagonal_int[i], diagonal_old[i])]
                                       , 'P': [(diagonal_non[i + N], diagonal_int[i + N], diagonal_old[i + N])]}

            else:
                oscillators[f"{i}"]['Q'].append((diagonal_non[i], diagonal_int[i], diagonal_old[i]))
                oscillators[f"{i}"]['P'].append((diagonal_non[i + N], diagonal_int[i + N], diagonal_old[i + N]))

        # Once the vector works
        # print(initial_state_vector)
        # a = vector_multiplication(S, initial_state_vector)
        # print(a)

        # The sanity check is performed
        if not sanity_check_1(J, S_noninteracting):
            print("The matrix S is not symplectic anymore.")
            break

    stop = timeit.default_timer()

    # Plotting
    if N != 1:
        for key, value in oscillators.items():
            plot_expectation(value["Q"], value["P"], TIMES, f"Expectation values for oscillator {key}")

    else:
        wigner_non = np.array(wigner_non)
        wigner_visualization(wigner_non[0], ax_range)

        fig = plt.figure()
        ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5), xlabel='x', ylabel='y', aspect='equal')

        # first image on screen
        cont = plt.contourf(ax_range, ax_range, wigner_non[0])

        wigner_animation(wigner_non, ax_range, fig, cont)

    print(f"\nRuntime: {(stop - start):.4f} s.")


if __name__ == '__main__':
    main()
