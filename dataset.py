import os
from math import sqrt, floor
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy import float64
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from scipy.io import arff
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_proj_root(cwd: Path):
    for root, dirs, files in os.walk(str(cwd)):
        if "README.md" in files:
            return cwd
        break
    cwd = cwd.parent
    return get_proj_root(cwd)


CWD = get_proj_root(Path().cwd())
CONTINENTS = ["africa", "asia", "europe", "north america",
              "south america", "antarctica", "australia", "oceania"]


class DataSet:
    def __init__(self, train_file_paths: List[Path], split_ratio: float = 0.8, merged: bool = False):
        """
        Creates a DataSet object using the given arff file.
        Calculates variables required to train the model and classify new instances.
        """

        # The name of the training file
        self.train_file: str = str(train_file_paths[0].parts[-1]) if len(train_file_paths) == 1 else "Merged"

        # Data set's name
        self.name = str(self.train_file).split(".")[0] if len(train_file_paths) == 1 else "Merged"

        # Is this a merging of the data sets
        self.merged = merged

        # Reads the training data from the train csv
        self.data: pd.DataFrame = pd.DataFrame(arff.loadarff(CWD.joinpath(train_file_paths[0]))[0]).iloc[:, 0:]

        # If this is a merging, drop specified columns
        if self.merged:
            for file_path in train_file_paths[1:]:
                self.data.drop(columns=[col for col in self.data.columns if "5" in col or "7" in col or "9" in col or "10" in col],
                               inplace=True)
                if "Adolescent" in str(file_path) or "Child" in str(file_path):
                    self.data.drop(columns=[col for col in self.data.columns if "8" in col], inplace=True)
                if "Adult" in str(file_path):
                    self.data.drop(columns=[col for col in self.data.columns if "1" in col], inplace=True)
                    cols = [col for col in self.data.columns if "8" not in col] + [col for col in self.data.columns if "8" in col]
                    self.data = self.data[[cols[-1]] + cols[:-1]]
                self.data.append(pd.DataFrame(arff.loadarff(CWD.joinpath(file_path))[0]).iloc[:, 0:])

        # Preprocess data
        self.preprocess()

        # Splits the training data into x and y
        self.X: pd.DataFrame = self.data.iloc[:, :-1]
        self.Y: pd.DataFrame = self.data.iloc[:, -1].T.astype(int)
        self.normalize()

        # Split data into train/test
        train_test = self.train_test_split(split_ratio)
        self.x_train: pd.DataFrame = train_test[0]
        self.x_test: pd.DataFrame = train_test[1]
        self.y_train: pd.DataFrame = train_test[2]
        self.y_test: pd.DataFrame = train_test[3]

        # Initialize models, accuracy, feature importances, and class predictions
        self.models: Dict = {"SVM": SVM, "RandomForest": RandomForest, "KNN": KNN}
        self.accuracy: Dict = {"SVM": 0, "RandomForest": 0, "KNN": 0}
        self.feature_importance: pd.DataFrame = pd.DataFrame(np.zeros((self.x_train.shape[1], 2)),
                                                             index=list(self.x_train.columns),
                                                             columns=["SVM", "RandomForest"])
        self.y_predict: pd.DataFrame = pd.DataFrame(np.zeros((self.y_test.shape[0], len(list(self.models.keys())))),
                                                    columns=list(self.models.keys()))

    def preprocess(self):
        """
        Preprocesses the data set
        """
        encoder = preprocessing.LabelEncoder()

        # Replace misspelled columns
        self.data.rename(columns={"austim": "autism", "contry_of_res": "country_of_res", "jundice": "jaundice"}, inplace=True)

        # Initialize list of continents
        continents = []

        # Iterate through the columns and process the data accordingly
        for col in self.data.columns:

            # Convert all byte-string values to strings
            if self.data[col].dtype.type != float64:
                self.data[col] = self.data[col].str.decode("utf-8")
                self.data[col] = self.data[col].str.lower()

            # Replace all missing values with NaN
            if "?" in list(self.data[col].values):
                self.data.loc[self.data[col] == "?", col] = np.nan

            # Replace all NaNs with the column's mode
            if self.data[col].isnull().values.any():
                self.remove_nans()
                mode = self.data[col].mode().values[0]
                self.data[col].fillna(mode, inplace=True)

            # Check for outlier age
            if col == "age":
                index = self.data[self.data[col] > 120].index
                if len(index) > 0:
                    self.data.drop(index=index, inplace=True)
                    self.data.reset_index(drop=True, inplace=True)

            if "country" in col:
                countries = [country for country in self.data["country_of_res"].values]
                for i in range(len(countries)):
                    if countries[i] == "americansamoa":
                        countries[i] = "american samoa"
                self.data["country_of_res"] = countries
                continents = []
                for country in self.data["country_of_res"].values:
                    if country not in CONTINENTS:
                        try:
                            continents.append(country_alpha2_to_continent_code(country_name_to_country_alpha2(country, "lower")))
                        except KeyError:
                            if "u.s." in country:
                                continents.append("NA")
                    else:
                        continents.append(country)

            if "ethnicity" in col or "country" in col or "age_desc" in col or "relation" in col or "continent" in col:
                self.data[col] = encoder.fit_transform(self.data[col])

            elif "Score" in col:
                self.data[col] = self.data[col].astype(int)

        self.data.insert(0, "continent", continents)
        self.data["continent"] = encoder.fit_transform(self.data["continent"])

        self.data.gender[self.data.gender == "m"] = 0
        self.data.gender[self.data.gender == "f"] = 1

        self.data.jaundice[self.data.jaundice == "no"] = 0
        self.data.jaundice[self.data.jaundice == "yes"] = 1

        self.data.autism[self.data.autism == "no"] = 0
        self.data.autism[self.data.autism == "yes"] = 1

        self.data.used_app_before[self.data.used_app_before == "no"] = 0
        self.data.used_app_before[self.data.used_app_before == "yes"] = 1

        self.data["Class/ASD"][self.data["Class/ASD"] == "no"] = 0
        self.data["Class/ASD"][self.data["Class/ASD"] == "yes"] = 1

        self.data.drop(columns=["continent"], inplace=True)

        for col in self.data.columns:
            self.data[col] = self.data[col].astype(float)

    def remove_nans(self):
        df_nans = self.data.isnull().sum().sum()

        if df_nans / len(self.data.index) < 0.1:
            self.data = self.data.dropna()

    def normalize(self):
        cols = ["age", "ethnicity", "country_of_res", "relation", "result"]

        for col in cols:
            if col == "result":
                self.X[self.X[col] >= 9.999] = 10.0
                self.X[col] = np.where(self.X[col].between(7.5, 9.999), 7.5, self.X[col])
                self.X[col] = np.where(self.X[col].between(5, 7.499), 5.0, self.X[col])
                self.X[col] = np.where(self.X[col].between(2.5, 4.999), 2.5, self.X[col])
                self.X[col] = np.where(self.X[col].between(0, 2.499), 0, self.X[col])
            mean_diff = self.X[col] - self.X[col].mean()
            std = self.X[col].std(ddof=0)
            self.X[col] = mean_diff / std if std != 0 else 0

    def train_test_split(self, percentage: float = 0.8):
        mask = np.random.rand(len(self.X)) < percentage
        return self.X[mask].reset_index(drop=True), self.X[~mask].reset_index(drop=True), \
               self.Y[mask].reset_index(drop=True), self.Y[~mask].reset_index(drop=True)

    def fit(self):
        k = floor(sqrt(len(self.x_train)))
        for model_name, model in self.models.items():
            self.models[model_name] = model(self.x_train, self.y_train, k)

            if model_name == "SVM":
                self.feature_importance[model_name] = pd.DataFrame(self.models[model_name].coef_.ravel(),
                                                                   index=self.x_train.columns)
            elif model_name == "RandomForest":
                self.feature_importance[model_name] = pd.DataFrame(self.models[model_name].feature_importances_,
                                                                   index=self.x_train.columns)

    def score(self):
        for model_name, model in self.models.items():
            self.accuracy[model_name] = self.models[model_name].score(self.x_test, self.y_test)

    def __repr__(self):
        return f"<DataSet Object> {self.train_file}"

    def __str__(self):
        return str(self.data)


def SVM(X: pd.DataFrame, y: pd.DataFrame, *args):
    svm = LinearSVC(tol=1e-5)
    svm.fit(X, y)
    return svm


def RandomForest(X: pd.DataFrame, y: pd.DataFrame, *args):
    rf = RandomForestClassifier(max_depth=2)
    rf.fit(X, y)
    return rf


def KNN(X: pd.DataFrame, y: pd.DataFrame, k: int):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    return knn


def fit_models(data_sets: List[DataSet]):
    for data_set in data_sets:
        data_set.fit()


def score_models(data_sets: List[DataSet]):
    for data_set in data_sets:
        data_set.score()


split = 0.7

ADOLESCENTS_PATH = CWD.joinpath("data", "Autism-Adolescent-Data Plus Description", "Autism-Adolescent-Data.arff")
ADULTS_PATH = CWD.joinpath("data", "Autism-Adult-Data Plus Description File", "Autism-Adult-Data.arff")
CHILDREN_PATH = CWD.joinpath("data", "Autism-Screening-Child-Data Plus Description", "Autism-Child-Data.arff")

Adolescents = DataSet([ADOLESCENTS_PATH], split_ratio=split)
Adults = DataSet([ADULTS_PATH], split_ratio=split)
Children = DataSet([CHILDREN_PATH], split_ratio=split)
Merged = DataSet([ADOLESCENTS_PATH, ADULTS_PATH, CHILDREN_PATH], split_ratio=split, merged=True)

ALL_DATA_SETS = [Adolescents, Adults, Children, Merged]
