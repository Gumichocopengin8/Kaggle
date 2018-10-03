import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

store = pd.read_csv("../../data/appleStore/appleStore.csv")


def lackTable(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    lack_table = pd.concat([null_val, percent], axis=1)
    lack_table_ren_columns = lack_table.rename(
        columns={0: 'num of lack', 1: '%'}
    )
    return lack_table_ren_columns


store.describe()
store_shape = store.shape
print(store_shape)
# plt.scatter(train.pickup_longitude.head(100), train.dropoff_longitude.head(100))
# plt.show()

