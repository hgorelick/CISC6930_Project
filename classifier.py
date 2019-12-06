from sys import platform
from typing import List

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
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
        prediction_result = np.unique(compare, return_counts=True)
        num_correct = prediction_result[1][1] if len(prediction_result[1]) > 1 else prediction_result[1][0]
        accuracy = num_correct / float(compare.shape[0])
        accuracies[data_set.name] = accuracy

    return data_sets, accuracies


fig_size = (22, 12)


def plot_accuracies(data_sets: List[DataSet], ensemble_accuracy: pd.DataFrame):
    data_sets_accuracies = pd.DataFrame({data_set.name: data_set.accuracy for data_set in data_sets})
    accuracies = data_sets_accuracies.append(ensemble_accuracy).T

    fig = accuracies.plot(figsize=fig_size, kind="bar", rot=0, width=0.7)
    title = f"Model Test Accuracies per Data Set"
    fig.set_title(title, fontsize=20)
    fig.tick_params(axis="x", labelsize=16)
    fig.set_ylabel("Accuracy", labelpad=20, fontsize=16)
    fig.tick_params(axis="y", direction="inout", which="both", labelsize=14)
    fig.yaxis.set_ticks(np.arange(0.0, 1.05, 0.05))
    fig.yaxis.set_ticks(np.arange(0.0, 1.01, 0.01), minor=True)

    plt.subplots_adjust(left=0.08, right=0.95)

    auto_label(fig)

    plt.grid(True, axis="y", color="#bfbfbf", linestyle=":")

    if platform == "win32":
        plt.get_current_fig_manager().window.state("zoomed")

    save_path = CWD.joinpath("figures", f"{title}.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importances_by_model(data_sets: List[DataSet], num_features: int = 5):
    names = [data_set.name.split("-")[1] if "-" in data_set.name else data_set.name for data_set in data_sets]

    svm = [data_set.feature_importance.nlargest(num_features, "SVM")["SVM"] for data_set in data_sets]
    rf = [data_set.feature_importance.nlargest(num_features, "RandomForest")["RandomForest"] for data_set in data_sets]

    feature_importances = {"SVM": pd.concat(svm, axis=1, keys=names, sort=False).T,
                           "RandomForest": pd.concat(rf, axis=1, keys=names, sort=False).T}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)

    for i, (name, features) in enumerate(feature_importances.items()):
        feature_importances[name].plot(ax=axes[i], title=name, kind="bar", rot=0, width=0.7, colormap=get_cmap("tab20"))

    axes[0].set_ylabel("Importance", labelpad=20, fontsize=16)

    setup(axes[0])
    auto_label(axes[0])

    setup(axes[1])
    auto_label(axes[1])

    title = f"Feature Importances by Model per Data Set"
    plt.suptitle(title, fontsize=24)

    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.97, top=0.9, wspace=0.11)

    if platform == "win32":
        plt.get_current_fig_manager().window.state("zoomed")

    save_path = CWD.joinpath("figures", f"{title}.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importances_by_data_set(data_sets: List[DataSet], num_features: int = 5):
    feature_importances = {data_set.name: data_set.feature_importance for data_set in data_sets}
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=fig_size)

    for i, (name, features) in enumerate(feature_importances.items()):
        feature_importances[name] = features.nlargest(num_features, ["SVM", "RandomForest"]).T
        feature_importances[name].plot(ax=axes[i], title=name, kind="bar", rot=0, width=0.7)

    setup(axes[0])
    auto_label(axes[0])

    setup(axes[1])
    auto_label(axes[1])

    title = "Feature Importances by Data Set per Model"
    plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.97)

    if platform == "win32":
        plt.get_current_fig_manager().window.state("zoomed")

    save_path = CWD.joinpath("figures", f"{title}.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def setup(ax, ymax=None):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(which="major", width=1.00)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", width=0.75)
    ax.tick_params(which="minor", length=2.5)
    ax.patch.set_alpha(0.0)
    if ymax is None:
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        ax.yaxis.set_ticks(np.arange(0.0, ymax, 0.05))
        ax.yaxis.set_ticks(np.arange(0.0, ymax, 0.01), minor=True)
    ax.grid(axis="y", linestyle="dotted", color="grey")


def auto_label(ax):
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

        if label_position > 0.1:
            ax.text(rect.get_x() + rect.get_width() / 2., label_position, "{0:.3f}".format(height), ha='center', va='bottom')


if __name__ == "__main__":
    data_sets, ensemble_accuracies = get_ensemble_accuracies(ALL_DATA_SETS)
    plot_accuracies(data_sets, ensemble_accuracies)
    plot_feature_importances_by_model(data_sets)
    # plot_feature_importances_by_data_set(data_sets)
