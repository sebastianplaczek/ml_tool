import yaml
import os
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from datetime import datetime


from sklearn.model_selection import KFold


class MLWorker:

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.read_config()
        self.create_folder_if_not_exists(self.params["charts_localisation"])
        self.create_folder_if_not_exists(
            self.params["charts_localisation"] + self.params["model_name"]
        )

    def create_folder_if_not_exists(self, folder_name):
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Utworzono folder: {folder_name}")
        else:
            print(f"Folder {folder_name} już istnieje.")

    def read_config(self):
        with open("params.yml", "r") as file:
            self.params = yaml.safe_load(file)

    def gini_normalized(y_test, y_pred):
        gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
        return gini(y_test, y_pred) / gini(y_test, y_actual)

    def model_selection(self):
        models = {
            "XGBClassifier": xgb.XGBClassifier(),
            "XGBRegressor": xgb.XGBRegressor(),
        }
        self.model = models[self.params["model_type"]]
        if self.params["model_params"] != "default":
            model.set_params(self.params["model_params"])

    def validation(self):
        now = str(datetime.now()).replace(" ", "_").replace(":", "")
        self.model_output_path = (
            self.params["charts_localisation"] + self.params["model_name"] + "//" + now
        )
        self.create_folder_if_not_exists(self.model_output_path)
        validations = {"cross_validation": self.cross_validation()}
        validations[self.params["validation_type"]]

    def cross_validation(self):
        X = self.X.values
        y = self.y.values
        kf = KFold(n_splits=self.params["validation_params"]["n_splits"])

        metrics = {
            "accuracy": accuracy_score,
            "recall": recall_score,
            "precision": precision_score,
            "gini": gini_normalized,
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
        }
        scores = {score: [] for score in self.params["model_scores"]}
        index = 0
        for train_index, test_index in kf.split(X):
            index += 1
            X_train, X_test = (
                X[train_index],
                X[test_index],
            )  # Zbiory treningowy i testowy cech
            y_train, y_test = (
                y[train_index],
                y[test_index],
            )  # Zbiory treningowy i testowy etykiet

            # Trenowanie modelu na zbiorze treningowym
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            # Ocena modelu na zbiorze testowym
            for score_name in self.params["model_scores"]:
                metrics_function = metrics[score_name]
                score = round(metrics_function(y_test, y_pred) * 100, 2)
                scores[score_name].append(score)
                print(f"{score_name} : {score}")

            # save chart with params in validation, every split on line

            # Rysowanie krzywej ROC
            if "roc" in self.params["save_charts"]:
                self.plot_roc(X_test, y_test, index)

            if "confusion_matrix" in self.params["save_charts"]:
                self.plot_conf_matrix(y_test, y_pred, index)

    def plot_roc(self, X_test, y_test, iteration):
        y_probs = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(self.model_output_path + "//" + f"roc_curve_{iteration}.png")

    def plot_conf_matrix(self, y_test, y_pred, iteration):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.savefig(self.model_output_path + "//" + f"confusion_matrix_{iteration}.png")
        # plt.show()

    def run(self):
        # co z regresja?

        self.model_selection()
        self.validation()
        t1 = "x"
