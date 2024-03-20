
import pandas as pd
import math
import numpy as np
from datatools import batch_data

class Network:
    def __init__(self, depth, epochs):
        self.depth = depth
        self.epochs = epochs
        self.batch_size = 50

    def sigmoid(self, z_vector):
        output = []
        for z in z_vector:
            sigmoid = 1 / (1 + math.e ** (-1 * z))
            output.append(sigmoid)
        return np.array(output)


    def train_network(self, data):
        all_weights = []
        average_gradients = []
        for i in range(self.depth):
            weight_matrix = np.ones((6, 7))
            average_matrix = np.zeros((6, 7))
            all_weights.append(weight_matrix)
            average_gradients.append(average_matrix)
        weight_matrix = np.ones((4, 7))
        all_weights.append(weight_matrix)
        average_matrix = np.zeros((4, 7))
        average_gradients.append(average_matrix)
        epoch_count = 0
        while epoch_count <= self.epochs:
            epoch_count += 1
            batches = batch_data(data, self.batch_size)
            for current_batch in batches:
                all_gradients = []
                for i in range(self.depth):
                    gradient_matrix = np.zeros((6, 7))
                    all_gradients.append(gradient_matrix)
                gradient_matrix = np.zeros((4, 7))
                all_gradients.append(gradient_matrix)
                for index, training_instance in current_batch.iterrows():
                    if training_instance['class'][0] == 'unacc':
                        y_actual = np.array([[1], [0], [0], [0]])
                    elif training_instance['class'][0] == 'acc':
                        y_actual = np.array([[0], [1], [0], [0]])
                    elif training_instance['class'][0] == 'good':
                        y_actual = np.array([[0], [0], [1], [0]])
                    else:
                        y_actual = np.array([[0], [0], [0], [1]])

                    x_vector = np.array([[training_instance['buying']], [training_instance['maint']],
                                        [training_instance['doors']], [training_instance['persons']],
                                        [training_instance['lug_boot']], [training_instance['safety']], [1]])

                    # forward pass through network
                    current_output = x_vector
                    for weight_layer in all_weights:
                        affine_output = np.matmul(weight_layer, current_output)
                        current_output = self.sigmoid(affine_output)
                        current_output = np.vstack([current_output, [1]])
                    y_predicted = current_output[:-1]

                    l2_loss = 0
                    l2_loss += (y_predicted[0] - y_actual[0]) ** 2
                    l2_loss += (y_predicted[1] - y_actual[1]) ** 2
                    l2_loss += (y_predicted[2] - y_actual[2]) ** 2
                    l2_loss += (y_predicted[3] - y_actual[3]) ** 2

                    # backward pass to calculate gradients
                    derivative_output = (np.subtract(y_predicted, y_actual)) * 2
                    derivative_output = derivative_output.T
