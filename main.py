import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
# import matplotlib;matplotlib.use("TkAgg")
import timeit
import networkx as nx
from random import randint, randrange

# The amount of oscillators and system temperature
N = 1
T = 1

# Defining a NxN zero matrix to help construct the other matrixes
zero = np.zeros((N, N))

# Defining the J matrix
J = np.block([[zero, np.identity(N)], [-1 * np.identity(N), zero]])

# The frequencies of the oscillators
frequencies = []
for i in range(N):
    frequencies.append(1)

frequencies = np.array(frequencies)

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


# Vectorizes the function and applys it to get the expectation values for P^2
E_P2_vector = np.vectorize(expectation_value_P2)
P2_expectation = E_P2_vector(frequencies)

# Creating the covariance matrix describing the initial state
cov_X_initial = np.block([[np.diag(Q2_expectation), zero], [zero, np.diag(P2_expectation)]])


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
        S = np.block([[K @ block_cos, K @ (inverse_freq_diag * block_sin)], [K @ (-freq_diag * block_sin), K @ block_cos]])
    else:
        S = np.block([[block_cos, inverse_freq_diag * block_sin], [-freq_diag * block_sin, block_cos]])
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


# For animation
def animation(Q, P):
    fig = plt.figure()
    anim = ani.FuncAnimation(fig, draw, len(Q), fargs=(Q, P), interval=1)

    plt.show()


# Plots the expectation values with respect to time
def plot_expectation(Q_values: list, P_values: list, times: list, label: str):
    """plt.clf()

    plt.plot(times, Q_values, label="Position")
    plt.plot(times, P_values, label="Momentum")

    plt.legend()
    plt.title(label)
    plt.xlabel("Time")
    plt.ylabel("Expectation value of the square")

    plt.show()"""

    """plt.figure(1)
    plt.subplot()
    plt.plot(times, Q_values[0], label="Q value")
    plt.plot(times, P_values[0], label="P value")
    plt.subplot(111)
    plt.plot(times, Q_values[1], label="Q value 1")"""

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


def main():
    # Starting the runtime -timer
    start = timeit.default_timer()

    # The graph containing N quantum mechanical harmonic oscillators
    G = nx.newman_watts_strogatz_graph(N, 2, 0.5)

    # Adding random weights
    for u, v in G.edges():
        G.edges[u, v]['weight'] = randint(1, 10)

    nx.draw_circular(G)
    plt.show()

    # Add at least one vibrator
    """G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, weight=2)"""

    # Need to add a way to add edges
    """G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 4, weight=1)
    G.add_edge(3, 4, weight=3)"""

    # Construction of matrix A
    A = matrix_A(G, frequencies)

    # The vector of eigenvalues and K -matrix
    eigenvalues, K = np.linalg.eigh(A)
    print(f"Diagonalized properly: {sanity_check_2(eigenvalues, K, A)}")

    # The simulated frames
    times = [i / 100 for i in range(200)]

    # A dict into which every oscillators' expectation values will be saved to
    oscillators = {}

    # Iterates the initial state in three different bases
    for t in times:
        # The corresponding symplectic matrix S in the non-interacting base for the value of t
        S_noninteracting = symplectic(t, K)
        new_cov_X_non = S_noninteracting @ cov_X_initial @ S_noninteracting.transpose()
        diagonal_non = np.diagonal(new_cov_X_non)

        # S in the interacting base
        S_interacting = symplectic(t, K, interacting=True)
        new_cov_X_int = S_interacting @ cov_X_initial @ S_interacting.transpose()
        diagonal_int = np.diagonal(new_cov_X_int)

        # S in the old base
        X = np.block([[K.transpose(), zero], [zero, K.transpose()]])
        S_old = X.transpose() @ S_noninteracting @ X
        new_cov_X_old = S_old @ cov_X_initial @ S_old.transpose()
        diagonal_old = np.diagonal(new_cov_X_old)

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

    # print(oscillators)

    # Plotting
    for key, value in oscillators.items():
        plot_expectation(value["Q"], value["P"], times, f"Expectation values for oscillator {key}")

    stop = timeit.default_timer()

    print(f"\nRuntime: {(stop - start):.4f} s.")


if __name__ == '__main__':
    main()
