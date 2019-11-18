from scipy.io import arff
from pathlib import Path
import os
import pandas as pd


def get_proj_root(cwd: Path):
    for root, dirs, files in os.walk(str(cwd)):
        if "requirements.txt" in files:
            return cwd
        break
    cwd = cwd.parent
    return get_proj_root(cwd)


CWD = get_proj_root(Path().cwd())


class DataSet:
    def __init__(self, train_file_path: Path):
        """
        Creates a NaiveBayesDataSet object using the given train csv.
        Calculates variables required to train the model and classify new instances.
        """

        # The name of the training file
        self.train_file: Path = train_file_path.parts[-1]

        # Reads the training data from the train csv
        self.train_data: pd.DataFrame = pd.DataFrame(arff.loadarff(CWD.joinpath(train_file_path))[0]).iloc[:, 1:]

        # Splits the training data into x and y
        self.x_train: pd.DataFrame = self.train_data.iloc[:, :-1]
        self.y_train: pd.DataFrame = self.train_data.iloc[:, -1].T

    def __repr__(self):
        return f"<DataSet Object> {self.train_file}"

    def __str__(self):
        return str(self.train_data)
