# main.py
# reads in data, splits into train, validation, and test data
# builds 3 decision tree models
# determines best model using cross validation and tests it on test data

import pandas as pd
from datatools import split_data, convert_data
import network

main_df = pd.read_csv("car.data", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
main_df = convert_data(main_df)
print(main_df.head())

train_data, test_val_data = split_data(main_df, 0.5)
val_data, test_data = split_data(test_val_data, 0.5)

test_net = network.Network(2, 1)
test_net.train_network(train_data)



