from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer # use for bag of words

def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

def dataTransformation(dataset_path):
    ln = LabelEncoder()
    cv = CountVectorizer()

    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    
    xtest = cv.fit_transform(dataset["processed_message"]).toarray() 
    ytest = ln.fit_transform(dataset["target"])
    return xtest, ytest

def predict(text, model, vectorizer):
    vectorized_text = vectorizer.transform(text)
    prediction = model.predict(vectorized_text)
    return prediction


def main():
    
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    test_data_path = "data/interim/testdata_with_features.csv"
    xtest, ytest = dataTransformation(test_data_path)

    y_pred = predict(xtest,model,vectorizer)

    # Evaluate performance
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)

if __name__ == "__main__":
    main()
