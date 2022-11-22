import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
# import matplotlib;matplotlib.use("TkAgg")
import timeit

# The amount of oscillators and system temperature
N = 1
T = 1

# The frequencies of the oscillators
frequencies = np.array([2])

# Initial state for the vector consisting of the expectation values of Q^2 and P^2
initial_state_vector = np.array([1, 0])


class Oscillator:
    # Class constructor
    def __init__(self, frequency: float):
        self.frequency = frequency

    # Calculates the expectation value of the operator Q_i squared
    def expectation_value_Q2(self):
        if T == 0:
            value = 0.5 / self.frequency

        else:
            value = (1 / (np.exp(self.frequency / T) - 1) + 0.5) / self.frequency

        return value

    # Calculates the expectation value of the operator P_i squared
    def expectation_value_P2(self):
        if T == 0:
            value = 0.5 * self.frequency

        else:
            value = (1 / (np.exp(self.frequency / T) - 1) + 0.5) * self.frequency

        return value


# Vectorizing the methods for
expectation_value_Q2_v = np.vectorize(Oscillator().expectation_value_Q2)
expectation_value_P2_v = np.vectorize(Oscillator().expectation_value_P2)

Q2_expectations = expectation_value_Q2_v(frequencies)
P2_expectations = expectation_value_P2_v(frequencies)

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
def symplectic(t):
    block_cos = np.diag(cosine_vector(frequencies, t))
    block_sin = np.diag(sine_vector(frequencies, t))

    freq_diag = np.diag(frequencies)
    inverse_freq_diag = np.linalg.inv(freq_diag)

    S = np.block([[block_cos, inverse_freq_diag * block_sin], [-freq_diag * block_sin, block_cos]])
    return S


def vector_multiplication(S, vector):
    return S @ vector


# Checks whether the symplectic matrix is really symplectic
def sanity_check(J, S):
    new_J_matrix = S @ J @ np.transpose(S)
    return np.allclose(J, new_J_matrix)


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
def plot_expectation(Q_values, P_values, times, label: str):
    plt.clf()

    plt.plot(times, Q_values, label="Position")
    plt.plot(times, P_values, label="Momentum")

    plt.legend()
    plt.title(label)
    plt.xlabel("Time")
    plt.ylabel("Expectation value of the square")

    plt.show()


def main():
    start = timeit.default_timer()

    """print(f"E[Q^2] = {Q2_expectation}")
    print(f"E[P^2] = {P2_expectation}")

    print(f"\nCovariance matrix at t=0:\n {cov_X_initial}")"""

    # Defining a NxN zero matrix to help construct the other matrixes
    zero = np.zeros((N, N))

    # Defining the J matrix
    J = np.block([[zero, np.identity(N)], [-1 * np.identity(N), zero]])

    oscillators = []
    Q2_expectations = []
    P2_expectations = []

    # Creates a list of oscillators
    for i in frequencies:
        oscillators.append(Oscillator(i))

    # Creating the covariance matrix describing the initial state
    cov_X_initial = np.block([[np.diag(Q2_expectations), zero], [zero, np.diag(P2_expectations)]])

    times = [i / 100 for i in range(600)]
    Q_values = []
    P_values = []

    # Iterates the initial state
    for t in times:
        # The corresponding symplectic matrix for the value of t
        S = symplectic(t)
        print(S)

        # New covariance matrix is calculated
        new_cov_X = S @ cov_X_initial @ S.transpose()

        # print(new_cov_X)
        print()

        Q, P = new_cov_X[0][0], new_cov_X[1][1]

        # The sanity check is excecuted
        if not sanity_check(J, S):
            print("The matrix S is not symplectic anymore.")
            break

        # Saves the expectation values of Q^2 and P^2
        Q_values.append(Q)
        P_values.append(P)

        print(initial_state_vector)

        a = vector_multiplication(S, initial_state_vector)
        print(a)


    # Plotting and possible animation
    # animation()
    plot_expectation(Q_values, P_values, times, "The time evolution of the squared momentum-\nand position operator's expectation values")
    plot_expectation()

    stop = timeit.default_timer()

    print(f"Time: {(stop - start):.4f} s.")


if __name__ == '__main__':
    main()
