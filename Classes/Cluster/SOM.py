import numpy as np

class SOM:
    def __init__(self, epoch, n_neurons, attributes_dimension):
        self.epochs = epoch
        self.n_neurons = n_neurons
        self.att_dimension = attributes_dimension
        self.neurons_matrix = np.zeros((n_neurons, n_neurons, attributes_dimension))
        self.hist_neurons_matrix = np.zeros((self.epochs, n_neurons, n_neurons, attributes_dimension))

    def cluster_data(self, data, init_radius, i_learning_rate):
        time_constant = len(data) / np.log(init_radius)
        for ep in range(self.epochs):
            for itera, att in enumerate(data):
                winner = get_winner(self.neurons_matrix, att)
                radius = decay_radius(init_radius, itera, time_constant)
                learning_rate = decay_learning_rate(i_learning_rate, itera, len(data))

                for i in range(self.neurons_matrix.shape[0]):
                    for j in range(self.neurons_matrix.shape[1]):
                        w_dist = np.sum((np.array([i, j]) - winner) ** 2)
                        w_dist = np.sqrt(w_dist)
                        if w_dist <= radius:
                            influence = calculate_influence(w_dist, radius)
                            self.neurons_matrix[i, j] = self.neurons_matrix[i, j] + learning_rate * influence * (
                                    att - self.neurons_matrix[i, j])
            self.hist_neurons_matrix[ep] = self.neurons_matrix

def get_winner(el1, el2):
    d = distance(el1, el2)
    best_element = np.argmin(d)
    return np.array([best_element // el1.shape[0], best_element % el1.shape[1]])

def distance(el1, el2):
    return np.sqrt(np.sum((el2 - el1) ** 2, axis=-1))


def calculate_influence(distance, radius):
    return np.exp(-distance / (2 * (radius ** 2)))


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)
