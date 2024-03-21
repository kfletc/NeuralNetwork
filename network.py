# network.py
# class to represent the actual neural network, has methods for training off of data input and passing data through
# the network to calculate outputs

import numpy as np
from datatools import batch_data, calculate_accuracy
import networkhelper

class Network:
    def __init__(self, depth, epochs):
        self.depth = depth
        self.epochs = epochs
        self.batch_size = 25
        self.rms_constant = 0.9
        self.learning_rate = 0.05
        self.all_weights = []
        for i in range(self.depth):
            weight_matrix = np.random.randn(6, 7) * np.sqrt(2 / 7)
            self.all_weights.append(weight_matrix)
        weight_matrix = np.random.randn(4, 7) * np.sqrt(2 / 7)
        self.all_weights.append(weight_matrix)
        self.tolerance = 50
        self.hold_all_weights = []
        self.hold_accuracy = 0
        self.dropping_accuracy = False
        self.dropping_count = 0

    # trains network with training data input until either max epochs is met or the validation set passed stops
    # increasing in accuracy
    def train_network(self, data, val_data):
        average_gradients = []
        for i in range(self.depth):
            average_matrix = np.zeros((6, 7))
            average_gradients.append(average_matrix)
        average_matrix = np.zeros((4, 7))
        average_gradients.append(average_matrix)

        epoch_count = 0
        prev_val_accuracy = 0
        val_accuracy = 0
        while epoch_count < self.epochs:
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
                    if training_instance['class'] == 'unacc':
                        y_actual = np.array([[1], [0], [0], [0]])
                    elif training_instance['class'] == 'acc':
                        y_actual = np.array([[0], [1], [0], [0]])
                    elif training_instance['class'] == 'good':
                        y_actual = np.array([[0], [0], [1], [0]])
                    else:
                        y_actual = np.array([[0], [0], [0], [1]])

                    x_vector = np.array([[training_instance['buying']], [training_instance['maint']],
                                        [training_instance['doors']], [training_instance['persons']],
                                        [training_instance['lug_boot']], [training_instance['safety']], [1]])

                    # forward pass through network
                    affine_output = [np.zeros((7, 1))]
                    y_output = [x_vector]
                    k = 0
                    for weight_layer in self.all_weights:
                        affine_output.append(np.matmul(weight_layer, y_output[k]))
                        y_output.append(networkhelper.sigmoid(affine_output[k + 1]))
                        y_output[k + 1] = np.vstack([y_output[k + 1], [1]])
                        k = k + 1
                    y_predicted = y_output[k][:-1]

                    l2_loss = 0
                    l2_loss += (y_predicted[0] - y_actual[0]) ** 2
                    l2_loss += (y_predicted[1] - y_actual[1]) ** 2
                    l2_loss += (y_predicted[2] - y_actual[2]) ** 2
                    l2_loss += (y_predicted[3] - y_actual[3]) ** 2

                    # backward pass to calculate gradients
                    derivative_output = (np.subtract(y_predicted, y_actual)) * 2
                    derivative_loss_output = derivative_output.T

                    for k in range(self.depth, -1, -1):
                        jacobian_affine = networkhelper.jacobian_sigmoid(y_output[k + 1][:-1])
                        derivative_loss_affine = np.matmul(derivative_loss_output, jacobian_affine)

                        # weight gradient calculation
                        weight_gradient = np.transpose(np.matmul(y_output[k], derivative_loss_affine))
                        weight_gradient = weight_gradient * (1 / self.batch_size)
                        all_gradients[k] = np.add(all_gradients[k], weight_gradient)

                        # calculate new derivative with respect to previous output layer
                        derivative_loss_output = np.matmul(derivative_loss_affine, self.all_weights[k][:, :-1])

                # calculate running average of gradients
                k = 0
                for average_gradient_matrix in average_gradients:
                    update_matrix = (1 - self.rms_constant) * (np.square(all_gradients[k]))
                    average_gradients[k] = np.add(self.rms_constant * average_gradient_matrix, update_matrix)
                    k = k + 1

                # update weight matrix
                prev_all_weights = self.all_weights
                k = 0
                for weight_matrix in self.all_weights:
                    average_gradient_matrix = average_gradients[k]
                    gradient_matrix = all_gradients[k]
                    for i in range(weight_matrix.shape[0]):
                        for j in range(weight_matrix.shape[1]):
                            step_size = self.learning_rate / np.sqrt(average_gradient_matrix[i][j])
                            self.all_weights[k][i][j] = self.all_weights[k][i][j] - step_size * gradient_matrix[i][j]
                    k = k + 1

            val_output = self.apply_network(val_data)
            val_accuracy = calculate_accuracy(val_output)
            if self.dropping_accuracy:
                if val_accuracy < self.hold_accuracy:
                    self.dropping_count += 1
                else:
                    self.dropping_count = 0
                    self.dropping_accuracy = False
                if self.dropping_count > self.tolerance:
                    self.all_weights = self.hold_all_weights
                    val_accuracy = self.hold_accuracy
                    break
            else:
                if prev_val_accuracy > val_accuracy > 0.75:
                    self.hold_all_weights = prev_all_weights
                    self.hold_accuracy = prev_val_accuracy
                    self.dropping_accuracy = True
                    self.dropping_count = 0
            prev_val_accuracy = val_accuracy

        # run training set through network and return training accuracy and validation accuracy
        train_output_data = self.apply_network(data)
        train_accuracy = calculate_accuracy(train_output_data)
        return train_accuracy, val_accuracy, epoch_count

    # passes data through the network to assign a predicted class to each data point
    def apply_network(self, data):
        return_set = data
        return_set['PredictedClass'] = "None"
        for index, training_instance in data.iterrows():
            x_vector = np.array([[training_instance['buying']], [training_instance['maint']],
                                 [training_instance['doors']], [training_instance['persons']],
                                 [training_instance['lug_boot']], [training_instance['safety']], [1]])
            affine_output = [np.zeros((7, 1))]
            y_output = [x_vector]
            k = 0
            for weight_layer in self.all_weights:
                affine_output.append(np.matmul(weight_layer, y_output[k]))
                y_output.append(networkhelper.sigmoid(affine_output[k + 1]))
                y_output[k + 1] = np.vstack([y_output[k + 1], [1]])
                k = k + 1
            y_predicted = (y_output[k][:-1]).T
            class_num = np.argmax(y_predicted)
            if class_num == 0:
                predicted_class = 'unacc'
            elif class_num == 1:
                predicted_class = 'acc'
            elif class_num == 2:
                predicted_class = 'good'
            else:
                predicted_class = 'vgood'
            return_set.at[index, 'PredictedClass'] = predicted_class
        return return_set
