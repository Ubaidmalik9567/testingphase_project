import pandas as pd
import pathlib
import sys
import yaml
from sklearn.model_selection import train_test_split

def read_data(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset

def split_data(data, split_datasetSize, seed):
    train_data,test_data = train_test_split(data, test_size=split_datasetSize, random_state=seed)
    return train_data,test_data

def save_dataset_file(filelocation,traindata,testdata):
    pathlib.Path(filelocation).mkdir(parents=True,exist_ok=True)
    traindata.to_csv(filelocation + "/train.csv", index=True)
    testdata.to_csv(filelocation + "/test.csv", index=True)


def main():

    curr_dir = pathlib.Path(__file__) 
    home_dir = curr_dir.parent.parent.parent

    rawdatasetpath_location = sys.argv[1]
    datasetpath_location = home_dir.as_posix() + rawdatasetpath_location

    paramsfile_location = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(paramsfile_location))["make_dataset"]

    save_splitingfile_pathLocation = home_dir.as_posix() + "/data/processed"

    dataset = read_data(datasetpath_location)
    train, test = split_data(dataset, params["split_datasetSize"], params["seed"])
    save_dataset_file(save_splitingfile_pathLocation, train, test)


if __name__ == "__main__":
    main()




