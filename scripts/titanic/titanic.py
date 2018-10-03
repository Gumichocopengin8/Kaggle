import pandas as pd
import numpy as np
from sklearn import tree

pd.set_option('display.max_columns', 20)

train = pd.read_csv("../../data/titanic/train.csv")
test = pd.read_csv("../../data/titanic/test.csv")

test.describe()

test_shape = test.shape
print(test_shape)


def lackTable(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    lack_table = pd.concat([null_val, percent], axis=1)
    lack_table_ren_columns = lack_table.rename(
        columns={0: 'num of lack', 1: '%'}
    )
    return lack_table_ren_columns


train.Age = train.Age.fillna(train.Age.median())
train.Embarked = train.Embarked.fillna("S")

train.Sex = train.Sex.replace("male", 0)
train.Sex = train.Sex.replace("female", 1)
train.Embarked = train.Embarked.replace("S", 0)
train.Embarked = train.Embarked.replace("C", 1)
train.Embarked = train.Embarked.replace("Q", 2)

test.Age = test.Age.fillna(test.Age.median())
test.Fare = test.Fare.fillna(test.Fare.median())
test.Sex = test.Sex.replace("male", 0)
test.Sex = test.Sex.replace("female", 1)
test.Embarked = test.Embarked.replace("S", 0)
test.Embarked = test.Embarked.replace("C", 1)
test.Embarked = test.Embarked.replace("Q", 2)


target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(features_one, target)
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

prediction = tree_one.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=["Survived"])
solution.to_csv("tree_one.csv", index_label=["PassengerId"])
