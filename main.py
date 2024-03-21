# main.py
# reads in data, splits into train, validation, and test data
# builds 3 neural network models with differing depths
# determines best model using holdout with a validation set
# calculates test set accuracy of the best model

import pandas as pd
from datatools import split_data, convert_data, calculate_accuracy
import network

main_df = pd.read_csv("car.data", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
main_df = convert_data(main_df)
print("Sample of data: ")
print(main_df.head())

print("\nSplitting data into 50% training, 25% validation, and 25% test data")
train_data, test_val_data = split_data(main_df, 0.5)
val_data, test_data = split_data(test_val_data, 0.5)

print("\nTraining network with 3 hidden layers")
network_1 = network.Network(3, 1000)
train_accuracy, val_accuracy_1, epochs = network_1.train_network(train_data, val_data)
print("Total Epochs: " + str(epochs))
print("training accuracy: " + str(train_accuracy))
print("validation accuracy: " + str(val_accuracy_1))

print("\nTraining network with 5 hidden layers")
network_2 = network.Network(5, 1000)
train_accuracy, val_accuracy_2, epochs = network_2.train_network(train_data, val_data)
print("Total Epochs: " + str(epochs))
print("training accuracy: " + str(train_accuracy))
print("validation accuracy: " + str(val_accuracy_2))

print("\nTraining network with 10 hidden layers")
network_3 = network.Network(10, 1000)
train_accuracy, val_accuracy_3, epochs = network_3.train_network(train_data, val_data)
print("Total Epochs: " + str(epochs))
print("training accuracy: " + str(train_accuracy))
print("validation accuracy: " + str(val_accuracy_3))

if val_accuracy_1 > val_accuracy_2 and val_accuracy_1 > val_accuracy_3:
    test_output = network_1.apply_network(test_data)
    hidden_layers = 3
elif val_accuracy_2 > val_accuracy_3 and val_accuracy_2 > val_accuracy_1:
    test_output = network_2.apply_network(test_data)
    hidden_layers = 5
else:
    test_output = network_3.apply_network(test_data)
    hidden_layers = 10

test_accuracy = calculate_accuracy(test_output)
print("\nBest network by validation accuracy has " + str(hidden_layers) + " hidden layers.")
print("Test set accuracy on network with " + str(hidden_layers) + " hidden layers: " + str(test_accuracy))

print("\ndone.")
