# datasplit.py
# contains methods for splitting data into train and test data as well as splitting for k-fold (k=4) cross validation

import pandas as pd

# split_data splits dataframe into 2 where the first has the given percentage of the data and the second has the rest
# data is stratified by class
def split_data(df, percentage):
    result_df_1 = pd.DataFrame(columns=df.columns)
    result_df_2 = pd.DataFrame(columns=df.columns)
    classes = df['class'].drop_duplicates()
    for c in classes:
        c_df = df[df['class'] == c]
        c_df_1 = c_df.sample(frac=percentage, replace=False)
        c_df_2 = c_df[~c_df.index.isin(c_df_1.index)]
        result_df_1 = pd.concat([result_df_1, c_df_1]).reset_index(drop=True)
        result_df_2 = pd.concat([result_df_2, c_df_2]).reset_index(drop=True)

    return result_df_1, result_df_2

# converts all data inputs to real values since they are all ordinal attributes
def convert_data(df):
    df['buying'] = df['buying'].replace(['vhigh', 'high', 'med', 'low'], [4.0, 3.0, 2.0, 1.0])
    df['maint'] = df['maint'].replace(['vhigh', 'high', 'med', 'low'], [4.0, 3.0, 2.0, 1.0])
    df['doors'] = df['doors'].replace(['2', '3', '4', '5more'], [1.0, 2.0, 3.0, 4.0])
    df['persons'] = df['persons'].replace(['2', '4', 'more'], [1.0, 2.0, 3.0])
    df['lug_boot'] = df['lug_boot'].replace(['small', 'med', 'big'], [1.0, 2.0, 3.0])
    df['safety'] = df['safety'].replace(['low', 'med', 'high'], [1.0, 2.0, 3.0])
    return df

# splits data into batches of given size and throws out remainder that doesn't fit the size
def batch_data(df, size):
    df = df.sample(frac=1)
    list_df = [df[i:i + size] for i in range(0, len(df), size)]
    list_df = list_df[:-1]  # last batch will be a different size in most cases
    return list_df

# given data that has predicted classes, return the accuracy of the predicted classes to the actual classes
def calculate_accuracy(df):
    total = df.shape[0]
    correct = 0
    for index, row in df.iterrows():
        if row['class'] == row['PredictedClass']:
            correct += 1
    return float(correct) / float(total)
