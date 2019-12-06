from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from dataset import DataSet, ALL_DATA_SETS, CWD


def get_ensemble_accuracies(data_sets: List[DataSet]):
    y = {data_set.name: pd.DataFrame() for data_set in data_sets}
    for data_set in data_sets:
        data_set.fit()
        data_set.score()
        for name, model in data_set.models.items():
            data_set.y_predict[name] = data_set.models[name].predict(data_set.x_test)
        y[data_set.name] = data_set.y_predict

    ensemble_y = {data_set_name: pd.DataFrame() for data_set_name, predictions in y.items()}

    for data_set_name, predictions in y.items():
        ensemble_y[data_set_name] = predictions.mode(axis=1)
        ensemble_y[data_set_name].columns = ["y"]

    accuracies = pd.DataFrame({data_set_name: {"Ensemble": 0} for data_set_name, predictions in y.items()})
    for data_set in data_sets:
        compare = ensemble_y[data_set.name]["y"] == data_set.y_test
        accuracy = np.unique(compare, return_counts=True)[1][1] / float(compare.shape[0])
        accuracies[data_set.name] = accuracy

    return data_sets, accuracies


def plot_accuracies(data_sets: List[DataSet], ensemble_accuracy: pd.DataFrame):
    figsize = (18, 12)

    data_sets_accuracies = pd.DataFrame({data_set.name: data_set.accuracy for data_set in data_sets})
    accuracies = data_sets_accuracies.append(ensemble_accuracy).T

    fig = accuracies.plot(figsize=figsize, kind="bar", rot=0, width=0.7)
    fig.set_title("Model Test Accuracies per Data Set", fontsize=20)
    fig.tick_params(axis="x", labelsize=16)
    fig.set_ylabel("Accuracy", labelpad=20, fontsize=16)
    fig.tick_params(axis="y", direction="inout", which="both", labelsize=14)
    fig.yaxis.set_ticks(np.arange(0.0, 1.05, 0.05))
    fig.yaxis.set_ticks(np.arange(0.0, 1.01, 0.01), minor=True)

    plt.subplots_adjust(left=0.08, right=0.95)

    autolabel(fig)

    plt.grid(True, axis="y", color="#bfbfbf", linestyle=":")
    plt.savefig(CWD.joinpath("figures", "Model Test Accuracies per Data Set.png"), bbox_inches="tight")

    plt.show()


def plot_feature_importances(data_sets: List[DataSet]):
    figsize = (18, 12)

    feature_importances = {data_set.name: data_set.feature_importance for data_set in data_sets}
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)

    for i, (name, features) in enumerate(feature_importances.items()):
        feature_importances[name] = features.nlargest(3, ["SVM", "RandomForest"]).T
        feature_importances[name].plot(ax=axes[i], title=name, kind="bar", rot=0, width=0.7)
        setup(axes[i], 1.21, 1.21)
        autolabel(axes[i])

    plt.suptitle("Feature Importances by Model per Data Set", fontsize=24)
    plt.subplots_adjust(left=0.05, right=0.95)

    plt.savefig(CWD.joinpath("figures", "Feature Importances by Model per Data Set.png"), bbox_inches="tight")
    plt.show()


def setup(ax, major_ymax, minor_ymax):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(which="major", width=1.00)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", width=0.75)
    ax.tick_params(which="minor", length=2.5)
    ax.patch.set_alpha(0.0)
    ax.yaxis.set_ticks(np.arange(0.0, major_ymax, 0.05))
    ax.yaxis.set_ticks(np.arange(0.0, minor_ymax, 0.01), minor=True)
    ax.grid(axis="y", linestyle="dotted", color="grey")


def autolabel(ax):
    rects = ax.patches

    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom + 0.05

    for rect in rects:
        height = rect.get_height()
        p_height = (height / y_height)

        if p_height > 0.95:
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position, "{0:.3f}".format(height), ha='center', va='bottom')


if __name__ == "__main__":
    data_sets, ensemble_accuracies = get_ensemble_accuracies(ALL_DATA_SETS)
    # plot_accuracies(data_sets, ensemble_accuracies)
    plot_feature_importances(data_sets)
