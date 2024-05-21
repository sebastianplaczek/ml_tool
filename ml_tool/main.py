import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

import mlflow
from mlflow.models import infer_signature

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

from .models import models_dict

# import helpers.models


class MLWorker:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.columns = list(X.columns)

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

    def gini_normalized(self, y_test, y_pred):
        gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
        return gini(y_test, y_pred) / gini(y_test, y_pred)

    def model_selection(self):

        self.model = models_dict[self.params["model_type"]]
        if self.params["model_params"] != "default":
            self.model.set_params(self.params["model_params"])

    def validation(self):
        self.now = str(datetime.now()).replace(" ", "_").replace(":", "")
        self.model_output_path = (
            self.params["charts_localisation"]
            + self.params["model_name"]
            + "//"
            + self.now
        )
        self.create_folder_if_not_exists(self.model_output_path)
        validations = {"cross_validation": self.cross_validation()}
        validations[self.params["validation_type"]]

    def cross_validation(self):
        X = self.X.values
        y = self.y.values
        kf = KFold(n_splits=self.params["validation_params"]["n_splits"])

        metrics = {
            # classification
            "accuracy": accuracy_score,
            "recall": recall_score,
            "precision": precision_score,
            "gini": self.gini_normalized,
            # regression
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
        }
        self.scores = {score: [] for score in self.params["model_scores"]}
        index = 0
        for train_index, test_index in kf.split(X):
            index += 1
            X_train, X_test = (
                X[train_index],
                X[test_index],
            )
            y_train, y_test = (
                y[train_index],
                y[test_index],
            )

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            for score_name in self.params["model_scores"]:
                metrics_function = metrics[score_name]
                score = round(metrics_function(y_test, y_pred) * 100, 2)
                self.scores[score_name].append(score)
                print(f"{score_name} : {score}")

            # save chart with params in validation, every split on line

            # plot roc ROC
            if "roc" in self.params["save_charts"]:
                self.plot_roc(X_test, y_test, index)

            if "confusion_matrix" in self.params["save_charts"]:
                self.plot_conf_matrix(y_test, y_pred, index)

            if "feature_importance" in self.params["save_charts"]:
                self.feature_importance(index)

        if "metrics" in self.params["save_charts"]:
            self.plot_metrics()

        if self.params["save_params"]:
            self.save_params()

        self.create_avg_metrics()

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

    def feature_importance(self, iteration):

        plt.figure(figsize=(10, 6))
        plt.barh(self.columns, self.model.feature_importances_)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance Plot")

        # Zapisz wykres do pliku graficznego
        plt.savefig(
            self.model_output_path + "//" + f"feature_importance_{iteration}.png"
        )

    def create_avg_metrics(self):
        self.avg_metrics = {}
        for score_name, score_list in self.scores.items():
            self.avg_metrics[score_name] = np.mean(score_list)

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        for score_name, score_list in self.scores.items():
            avg = np.mean(score_list)
            splits = [str(x) for x in range(len(score_list))]
            plt.scatter(splits, score_list, label=f"{score_name}")
            plt.plot(splits, [avg for x in splits], label=f"{score_name}_avg")
        plt.xlabel("score")
        plt.ylabel("validation split number")
        plt.title(f"Validation metrics")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(self.model_output_path + "//" + "metrics.png")

    def save_params(self):
        self.params["features"] = self.columns
        filepath = self.model_output_path + "//" + "params.yml"
        with open(filepath, "w") as file:
            yaml.dump(self.params, file)

        print(f"Params saved")

    def save_excel(self):
        filename = "models.xlsx"
        df = pd.DataFrame()
        df.loc[0, "time"] = self.now
        df.loc[0, "features"] = str(self.params["features"])
        for score_name, score_list in self.scores.items():
            df.loc[0, score_name] = np.mean(score_list)
        df.loc[0, "model_params"] = str(self.params["model_params"])

        if os.path.exists(filename):
            df_models = pd.read_excel(filename)
            final = pd.concat([df_models, df], ignore_index=True)
            final.to_excel(filename)
            print("Model data saved")
            models_xlsx = final

        else:
            df.to_excel(filename)
            models_xlsx = df

        # resize columns
        with pd.ExcelWriter(filename) as writer:
            models_xlsx.to_excel(writer, index=False, sheet_name="Sheet1")
            sheet = writer.sheets["Sheet1"]

            for column_cells in sheet.columns:
                length = max(len(str(cell.value)) for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = (
                    length + 2
                )
            print("Moded data reshaped")

    def run_mlflow(self):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment(f"{self.params['model_name']}")
        with mlflow.start_run():
            if self.params["model_params"] != "default":
                mlflow.log_params(self.params["model_params"])

            mlflow.log_metrics(self.avg_metrics)
            signature = infer_signature(self.X, self.model.predict(self.X))
            model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature,
                # input_example=self.X,
                # registered_model_name="tracking-quickstart",
            )

    def run(self):

        self.model_selection()
        self.validation()
        if self.params["mlflow"]:
            self.run_mlflow()
        self.save_excel()


# dla regresjii zrobic print wykresu predictions vs real data na tescie
