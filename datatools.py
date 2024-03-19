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

