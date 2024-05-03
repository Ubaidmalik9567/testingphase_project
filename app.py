# from logger import logging

# if __name__ == "__main__":
#     logging.info("logging Start the application")
 
# import pandas as pnd
# dataset = pnd.read_csv("C:\\Users\\numl-\\testingphase_project\\data\\processed\\train.csv")  
# print(dataset.head(2))
# dataset.drop(columns=["COL 3","COL 4","COL 5"],inplace=True)
# dataset.rename(columns={"Unnamed: 0":"index","COL 1":"target","COL 2":"message"},inplace=True)
# print(dataset.head(2))
# dataset.rename(columns={"Unnamed: 0":"index","COL 1":"target","COL 2":"message"},inplace=True)
# print(dataset.head(2))
# print("cicd run")
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)