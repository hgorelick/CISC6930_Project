import os
from copy import deepcopy
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
    def __init__(self, train_file_paths: List[Path], split_ratio: float = 0.8):
        """
        Creates a DataSet object using the given arff file.
        Calculates variables required to train the model and classify new instances.
        """

        # The name of the training file
        self.train_file: str = str(train_file_paths[0].parts[-1]) if len(train_file_paths) == 1 else "Merged"

        # Data set's name
        self.name = str(self.train_file).split(".")[0] if len(train_file_paths) == 1 else "Merged"

        # Reads the training data from the train csv
        self.data: pd.DataFrame = pd.DataFrame(arff.loadarff(CWD.joinpath(train_file_paths[0]))[0]).iloc[:, 0:]

        if len(train_file_paths) > 1:
            for file_path in train_file_paths[1:]:
                self.data.append(pd.DataFrame(arff.loadarff(CWD.joinpath(file_path))[0]).iloc[:, 0:])
        self.preprocess()

        # Splits the training data into x and y
        self.X: pd.DataFrame = self.data.iloc[:, :-1]
        self.Y: pd.DataFrame = self.data.iloc[:, -1].T.astype(int)
        self.normalize()

        train_test = self.train_test_split(split_ratio)
        self.x_train: pd.DataFrame = train_test[0]
        self.x_test: pd.DataFrame = train_test[1]
        self.y_train: pd.DataFrame = train_test[2]
        self.y_test: pd.DataFrame = train_test[3]

        self.models: Dict = {"SVM": SVM, "RandomForest": RandomForest, "KNN": KNN}
        self.accuracy: Dict = {"SVM": 0, "RandomForest": 0, "KNN": 0}
        self.feature_importance: pd.DataFrame = pd.DataFrame(np.zeros((self.x_train.shape[1], 2)),
                                                             index=list(self.x_train.columns),
                                                             columns=["SVM", "RandomForest"])
        self.y_predict: pd.DataFrame = pd.DataFrame(np.zeros((self.y_test.shape[0], len(list(self.models.keys())))),
                                                    columns=list(self.models.keys()))
        # self.fit()
        # self.score()

    def preprocess(self):
        encoder = preprocessing.LabelEncoder()
        self.data.rename(columns={"jundice": "jaundice", "austim": "autism", "contry_of_res": "country_of_res"},
                         inplace=True)
        continents = []

        for col in self.data.columns:
            if self.data[col].dtype.type != float64:
                self.data[col] = self.data[col].str.decode("utf-8")
                self.data[col] = self.data[col].str.lower()

            if "?" in list(self.data[col].values):
                mode = self.data[col].mode().values[0]
                self.data.loc[self.data[col] == "?", col] = mode

            if self.data[col].isnull().values.any():
                mode = self.data[col].mode().values[0]
                self.data[col].fillna(mode, inplace=True)

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

                # continents = [country_alpha2_to_continent_code(country_name_to_country_alpha2(country, "lower"))
                #               if country not in CONTINENTS else country for country in self.train_data["country_of_res"].values]

            if "ethnicity" in col or "country" in col or "age_desc" in col or "relation" in col or "continent" in col:
                self.data[col] = encoder.fit_transform(self.data[col])

            elif "Score" in col:
                self.data[col] = self.data[col].astype(int)

        self.data.insert(16, "continent", continents)
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

        self.data.drop(columns=["continent", "result"], inplace=True)

        for col in self.data.columns:
            self.data[col] = self.data[col].astype(float)
        #
        # self.train_data.reset_index(drop=True, inplace=True)

    def normalize(self):
        for col in ["age", "ethnicity", "country_of_res", "relation"]:
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
Merged = DataSet([ADOLESCENTS_PATH, ADULTS_PATH, CHILDREN_PATH], split_ratio=split)

ALL_DATA_SETS = [Adolescents, Adults, Children, Merged]


if __name__ == "__main__":
    d = DataSet(CWD.joinpath("data", "Autism-Adolescent-Data Plus Description", "Autism-Adolescent-Data.arff"))
    print(d)
    pass
