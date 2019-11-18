"""
Author:     Henry Gorelick
Class:      CISC 6525
Assignment: HW 1
"""

from typing import List

import numpy as np


class DataSet:
    def __init__(self, train: str, test: str):
        """
        Creates a DataSet object using the given train/test csvs
        """
        # The name of the training file
        self.train_file: str = train

        # Reads the training data from the train csv and then adds the x0 column (all = 1)
        self.train_data: np.ndarray = np.insert(np.genfromtxt("./datasets/" + train, delimiter=',', skip_header=1),
                                                0, 1, axis=1)

        # Splits the training data into x and y
        self.x_train: np.ndarray = np.array(self.train_data[:, :-1])
        self.y_train: np.ndarray = np.array(self.train_data[:, -1]).T

        # Creates a dictionary for saving MSEs as values of a desired key.
        # The key used most in this assignment is lambda. Allows for easily plotting mse.keys vs mse.values
        self.train_mse = {}

        # All test members are the same as above
        self.test_file: str = test
        self.test_data: np.ndarray = np.insert(np.genfromtxt("./datasets/" + test, delimiter=',', skip_header=1),
                                               0, 1, axis=1)

        self.x_test: np.ndarray = np.array(self.test_data[:, :-1])
        self.y_test: np.ndarray = np.array(self.test_data[:, -1]).T

        self.test_mse = {}

    @property
    def train_min_mse(self):
        """
        Returns the minimum MSE for the training data
        """
        return min(list(self.train_mse.values()))

    @property
    def train_min_mse_lambda(self):
        """
        Returns the lambda for the minimum MSE of the training data
        """
        return min(self.train_mse, key=self.train_mse.get)

    @property
    def test_min_mse(self):
        """
        Returns the minimum MSE for the test data
        """
        return min(list(self.test_mse.values()))

    @property
    def test_min_mse_lambda(self):
        """
        Returns the lambda for the minimum MSE of the test data
        """
        return min(self.test_mse, key=self.test_mse.get)

    @property
    def train_result_str(self):
        return "\tTraining\n" \
               "\t--------\n" \
               "\t   MSE: {}\n" \
               "\tlambda: {}\n\n".format(self.train_min_mse, self.train_min_mse_lambda)

    @property
    def test_result_str(self):
        return "\t  Test  \n" \
               "\t--------\n" \
               "\t   MSE: {}\n" \
               "\tlambda: {}\n".format(self.test_min_mse, self.test_min_mse_lambda)

    @property
    def result_str(self):
        return "Minimum MSE values:\n" + self.train_result_str + self.test_result_str


class PlottedDataSet(DataSet):
    def __init__(self, train: str, test: str):
        """
        Inherits from DataSet but adds some extra class members to simplify plotting
        """
        super().__init__(train, test)

        self.title: str = "{} vs. {}".format(train.split('.')[0], test.split('.')[0])
        self.y_train_legend = train.split('.')[0]
        self.y_test_legend = test.split('.')[0]


# region Predefined Lists of DataSets
q1_data_sets: List[PlottedDataSet] = [PlottedDataSet("train-100-10.csv", "test-100-10.csv"),
                                      PlottedDataSet("train-100-100.csv", "test-100-100.csv"),
                                      PlottedDataSet("train-1000-100.csv", "test-1000-100.csv"),
                                      PlottedDataSet("train-50(1000)-100.csv", "test-1000-100.csv"),
                                      PlottedDataSet("train-100(1000)-100.csv", "test-1000-100.csv"),
                                      PlottedDataSet("train-150(1000)-100.csv", "test-1000-100.csv")]
q1b_data_sets: List[PlottedDataSet] = [PlottedDataSet("train-50(1000)-100.csv", "test-1000-100.csv"),
                                       PlottedDataSet("train-100(1000)-100.csv", "test-1000-100.csv"),
                                       PlottedDataSet("train-150(1000)-100.csv", "test-1000-100.csv")]

q2_data_sets: List[DataSet] = [DataSet("train-100-10.csv", "test-100-10.csv"),
                               DataSet("train-100-100.csv", "test-100-100.csv"),
                               DataSet("train-1000-100.csv", "test-1000-100.csv"),
                               DataSet("train-50(1000)-100.csv", "test-1000-100.csv"),
                               DataSet("train-100(1000)-100.csv", "test-1000-100.csv"),
                               DataSet("train-150(1000)-100.csv", "test-1000-100.csv")]
# endregion
