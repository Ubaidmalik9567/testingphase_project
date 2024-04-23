import sys
import pandas as pd
import pathlib
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def remove_columns(traindata,testdata):
    
    traindata = pd.read_csv(traindata)
    testdata = pd.read_csv(testdata)

    traindata.drop(columns=["COL 3","COL 4","COL 5"],inplace=True)
    traindata.rename(columns={"Unnamed: 0":"index","COL 1":"target","COL 2":"message"},inplace=True)
    traindata.drop_duplicates(keep="first",inplace=True)


    testdata.drop(columns=["COL 3","COL 4","COL 5"],inplace=True)
    testdata.rename(columns={"Unnamed: 0":"index","COL 1":"target","COL 2":"message"},inplace=True)
    testdata.drop_duplicates(keep="first",inplace=True)

    return traindata,testdata


def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    stemming = PorterStemmer()
    for i in text:
        y.append(stemming.stem(i))
    return " ".join(y)


def feature_engineering(dataset):
    
    dataset['processed_message'] = dataset['message'].apply(text_preprocessing) # Text preprocessing
    # Additional features
    
    # dataset['no_of_characters'] = dataset['message'].apply(len)
    # dataset['no_of_words'] = dataset['message'].apply(lambda x: len(nltk.word_tokenize(x)))
    # dataset['no_of_sentences'] = dataset['message'].apply(lambda x: len(nltk.sent_tokenize(x)))
    
    dataset['no_of_characters'] = dataset['processed_message'].apply(len)
    dataset['no_of_words'] = dataset['processed_message'].apply(lambda x: len(nltk.word_tokenize(x)))
    dataset['no_of_sentences'] = dataset['processed_message'].apply(lambda x: len(nltk.sent_tokenize(x)))
    
    # return dataset[['processed_message', 'no_of_characters', 'no_of_words', 'no_of_sentences',"target"]]
    return dataset.drop(columns=["message"])


def saveFeatureEngineering_files(trainingfile,testingfile,savingPathlocation):
    
    pathlib.Path(savingPathlocation).mkdir(parents=True, exist_ok=True)
    trainingfile.to_csv(savingPathlocation + "/traindata_with_features.csv", index=False)
    testingfile.to_csv(savingPathlocation + "/testdata_with_features.csv", index=False)


def main():

    curr_dir = pathlib.Path(__file__) 
    home_dir = curr_dir.parent.parent.parent

    datasetfiles_path = sys.argv[1]
    dataset_path = home_dir.as_posix() + datasetfiles_path

    
    save_csvfile_path = sys.argv[2]
    processed_saveingfilepath_location = home_dir.as_posix() + save_csvfile_path

    traindata_path = dataset_path + "/train.csv"
    testdata_path = dataset_path + "/test.csv"

    train, test = remove_columns(traindata_path, testdata_path)
    processedtrain_data = feature_engineering(train)
    processedtest_data = feature_engineering(test)

    saveFeatureEngineering_files(processedtrain_data,processedtest_data,processed_saveingfilepath_location)


if __name__ == "__main__":
    main()