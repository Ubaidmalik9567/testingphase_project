from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer # use for bag of words
import pickle

def dataTransformation(dataset_path):
    ln = LabelEncoder()
    cv = CountVectorizer() 
    minmax = MinMaxScaler()

    dataset = pd.read_csv(dataset_path)
    dataset.dropna(subset=["processed_message"], inplace=True)

    processed_message_vectorized = cv.fit_transform(dataset["processed_message"]).toarray()
    processed_message_df = pd.DataFrame(processed_message_vectorized, columns=[f"feature_{i}" for i in range(processed_message_vectorized.shape[1])])
    
    dataset[['no_of_characters','no_of_words']] = minmax.fit_transform(dataset[['no_of_characters','no_of_words']])  
    dataset["target"] = ln.fit_transform(dataset["target"])
    
    x_train = pd.concat([processed_message_df, dataset[['no_of_characters', 'no_of_words', 'no_of_sentences']].reset_index(drop=True)], axis=1)
    y_train = dataset["target"]

    # xtrain = cv.fit_transform(dataset["processed_message"]).toarray()
    # ytrain = ln.fit_transform(dataset["target"])

    return x_train, y_train, cv

def train_model(x_train,y_train):
    lr = LogisticRegression() 
    lr.fit(x_train,y_train)
     # Print the number of features used by the model
    print(f"Number of features used by Logistic Regression: {lr.coef_.shape[1]}")
    
    return lr


def saveModel(trained_model, cv, pathLocation):
    
    pathlib.Path(pathLocation).mkdir(parents=True, exist_ok=True)
    with open(pathLocation + "/model.pkl", "wb") as model_file:
        pickle.dump(trained_model, model_file)

    with open(pathLocation + "/vectorizer.pkl", "wb") as cv_file:
        pickle.dump(cv, cv_file)


def main():
    curr_dir = pathlib.Path(__file__) 
    home_dir = curr_dir.parent.parent.parent

    path = sys.argv[1]
    datasetpath_location = home_dir.as_posix() + path 
    datasetpath_location = datasetpath_location + "/traindata_with_features.csv"
    modelsaving_pathLocation = home_dir.as_posix() + "/models"

    x, y, cv = dataTransformation(datasetpath_location)
    model = train_model(x, y)
    saveModel(model, cv, modelsaving_pathLocation)

if __name__ == "__main__":
    main()