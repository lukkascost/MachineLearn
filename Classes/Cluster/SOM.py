import numpy as np
import gzip
import pickle as pk


class SOM:
    def __init__(self, epoch, n_neurons, attributes_dimension):
        self.epochs = epoch
        self.n_neurons = n_neurons
        self.att_dimension = attributes_dimension
        self.neurons_matrix = np.zeros((n_neurons, n_neurons, attributes_dimension))
        self.hist_neurons_matrix = []
        self.hist_error = []
        self.neurons_labels = np.zeros((n_neurons, n_neurons, 1))

    def cluster_data(self, data, init_radius, i_learning_rate):
        time_constant = 0.0002
        time_constant2 = 0.0002

        print("Iniciando Treinamento...")
        continues = True
        ep = 0
        while continues:
            print("\n\n\n\nIniciando Epoca ", ep)
            for itera, att in enumerate(data):
                winner = get_winner(self.neurons_matrix, att)
                radius = decay_radius(init_radius, itera + (len(data) * ep), time_constant)
                learning_rate = decay_learning_rate(i_learning_rate, itera + (len(data) * ep), time_constant2)
                for i in range(self.neurons_matrix.shape[0]):
                    for j in range(self.neurons_matrix.shape[1]):
                        w_dist = np.sum((np.array([i, j]) - winner) ** 2)
                        w_dist = np.sqrt(w_dist)
                        if w_dist <= radius:
                            influence = calculate_influence(w_dist, radius)
                            self.neurons_matrix[i, j] = self.neurons_matrix[i, j] + learning_rate * influence * (
                                    att - self.neurons_matrix[i, j])

            self.hist_neurons_matrix.append(self.neurons_matrix)
            print(radius, learning_rate)
            print("Epoca ", ep, "encerrada")
            print("Iniciando calculo de erro de quantizacao...")
            self.calculate_error(ep, data)
            if ep == 0:
                ep += 1
                continue
            print("Condicao de parada", abs(self.hist_error[ep - 1] - self.hist_error[ep]))
            if abs(self.hist_error[ep - 1] - self.hist_error[ep]) < 0.00005:# or ep > 30:
                continues = False
            ep += 1

    def calculate_error(self, ep, data):
        self.hist_error.append(0)
        for itera, att in enumerate(data):
            winner = get_winner(self.neurons_matrix, att)
            self.hist_error[ep] += np.sum((att - self.hist_neurons_matrix[ep][winner[0], winner[1]]) ** 2)
        self.hist_error[ep] /= data.shape[0]
        print("Erro de quantizacao encontrado.")
        print(self.hist_error[ep])

    def save(self, filename, protocol=0):
        """
        Saves a compressed object to disk
        Parameter filename:
        Optional Parameter:
        """
        file_res = gzip.GzipFile(filename, 'wb')
        file_res.write(pk.dumps(self, protocol))
        file_res.close()

    @staticmethod
    def load(filename):
        """
        Loads a compressed object from disk
        Parameter filename:
        """
        file_res = gzip.GzipFile(filename, 'rb')
        class_file = b''
        while True:
            data = file_res.read()
            if data == b'':
                break
            class_file += data
        result = pk.loads(class_file)
        file_res.close()
        return result


def get_winner(el1, el2):
    d = distance(el1, el2)
    best_element = np.argmin(d)
    return np.array([best_element // el1.shape[0], best_element % el1.shape[1]])


def distance(el1, el2):
    return np.sqrt(np.sum((el2 - el1) ** 2, axis=-1))


def calculate_influence(distance, radius):
    return np.exp(-distance / (2 * (radius ** 2)))


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i * time_constant)


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i * n_iterations)
