import os
from pathlib import Path
from typing import List, Dict
from math import sqrt, floor

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
    def __init__(self, train_file_path: Path, split_ratio: float = 0.8):
        """
        Creates a DataSet object using the given arff file.
        Calculates variables required to train the model and classify new instances.
        """

        # The name of the training file
        self.train_file: Path = train_file_path.parts[-1]

        # Reads the training data from the train csv
        self.train_data: pd.DataFrame = pd.DataFrame(arff.loadarff(CWD.joinpath(train_file_path))[0]).iloc[:, 0:]
        self.preprocess()

        # Splits the training data into x and y
        self.X: pd.DataFrame = self.train_data.iloc[:, :-1]
        self.Y: pd.DataFrame = self.train_data.iloc[:, -1].T.astype(int)
        self.normalize()

        train_test = self.train_test_split(split_ratio)
        self.x_train: pd.DataFrame = train_test[0]
        self.x_test: pd.DataFrame = train_test[1]
        self.y_train: pd.DataFrame = train_test[2]
        self.y_test: pd.DataFrame = train_test[3]

        self.models: Dict = {"SVM": SVM, "RandomForest": RandomForest, "KNN": KNN}
        self.feature_importance: Dict = {"SVM": [], "RandomForest": [], "KNN": []}
        self.accuracy: Dict = {"SVM": 0, "RandomForest": 0, "KNN": 0}

    def preprocess(self):
        encoder = preprocessing.LabelEncoder()
        self.train_data.rename(columns={"jundice": "jaundice", "austim": "autism", "contry_of_res": "country_of_res"},
                               inplace=True)
        continents = []

        for col in self.train_data.columns:
            if self.train_data[col].dtype.type != float64:
                self.train_data[col] = self.train_data[col].str.decode("utf-8")

                self.train_data[col] = self.train_data[col].str.lower()

            if "?" in list(self.train_data[col].values):
                mode = self.train_data[col].mode().values[0]
                self.train_data.loc[self.train_data[col] == "?", col] = mode

            if "country" in col:
                countries = [country for country in self.train_data["country_of_res"].values]
                for i in range(len(countries)):
                    if countries[i] == "americansamoa":
                        countries[i] = "american samoa"
                self.train_data["country_of_res"] = countries
                continents = []
                for country in self.train_data["country_of_res"].values:
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
                self.train_data[col] = encoder.fit_transform(self.train_data[col])

            elif "Score" in col:
                self.train_data[col] = self.train_data[col].astype(int)

        self.train_data.insert(16, "continent", continents)
        self.train_data["continent"] = encoder.fit_transform(self.train_data["continent"])

        self.train_data.gender[self.train_data.gender == "m"] = 0
        self.train_data.gender[self.train_data.gender == "f"] = 1

        self.train_data.jaundice[self.train_data.jaundice == "no"] = 0
        self.train_data.jaundice[self.train_data.jaundice == "yes"] = 1

        self.train_data.autism[self.train_data.autism == "no"] = 0
        self.train_data.autism[self.train_data.autism == "yes"] = 1

        self.train_data.used_app_before[self.train_data.used_app_before == "no"] = 0
        self.train_data.used_app_before[self.train_data.used_app_before == "yes"] = 1

        self.train_data["Class/ASD"][self.train_data["Class/ASD"] == "no"] = 0
        self.train_data["Class/ASD"][self.train_data["Class/ASD"] == "yes"] = 1

        self.train_data.drop(columns=["continent", "result"], inplace=True)

    def normalize(self):
        for col in ["age", "ethnicity", "country_of_res", "relation"]:
            mean_diff = self.X[col] - self.X[col].mean()
            std = self.X[col].std(ddof=0)
            self.X[col] = mean_diff / std if std != 0 else 0

    def train_test_split(self, percentage: float = 0.8):
        mask = np.random.rand(len(self.X)) < percentage
        return self.X[mask], self.X[~mask], self.Y[mask], self.Y[~mask]

    def fit(self):
        k = floor(sqrt(len(self.x_train)))
        for model_name, model in self.models.items():
            self.models[model_name] = model(self.x_train, self.y_train, k)
            self.feature_importance[model_name] = pd.DataFrame(self.models[model_name].coef_.ravel(), columns=self.x_train.columns) \
                if model_name == "SVM" \
                else pd.DataFrame(self.models[model_name].feature_importances_, columns=self.x_train.columns)\
                if model_name != "KNN" else []

    def score(self):
        for model_name, model in self.models.items():
            self.accuracy[model_name] = self.models[model_name].score(self.x_test, self.y_test)

    def __repr__(self):
        return f"<DataSet Object> {self.train_file}"

    def __str__(self):
        return str(self.train_data)


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


split = 0.7
Adolescents = DataSet(CWD.joinpath("data", "Autism-Adolescent-Data Plus Description", "Autism-Adolescent-Data.arff"), split_ratio=split)
Adults = DataSet(CWD.joinpath("data", "Autism-Adult-Data Plus Description File", "Autism-Adult-Data.arff"), split_ratio=split)
Children = DataSet(CWD.joinpath("data", "Autism-Screening-Child-Data Plus Description", "Autism-Child-Data.arff"), split_ratio=split)

if __name__ == "__main__":
    d = DataSet(CWD.joinpath("data", "Autism-Adolescent-Data Plus Description", "Autism-Adolescent-Data.arff"))
    print(d)
    pass
