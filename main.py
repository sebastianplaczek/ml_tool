import yaml
import os
import xgboost as xgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)


class MLWorker:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def read_config(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, "config")
        file_path = os.path.join(config_path, "params.yml")

        with open(file_path, "r") as file:
            self.params = yaml.safe_load(file)

    def model_selection(self):
        models = {"XGBClassifier": xgb.XGBClassifier()}
        self.model = models[self.params["model_type"]]
        if self.params["model_params"] != "default":
            model.set_params(self.params["model_params"])

    def validation(self):
        validations = {"cross_validation", self.cross_validation()}
        validations[self.params["validation_type"]]

    def cross_validation(self):
        X = self.X
        y = self.y
        kf = KFold(n_splits=self.params["validation_params"]["n_splits"])

        for train_index, test_index in kf.split(X):
            X_train, X_test = (
                X[train_index],
                X[test_index],
            )  # Zbiory treningowy i testowy cech
            y_train, y_test = (
                y[train_index],
                y[test_index],
            )  # Zbiory treningowy i testowy etykiet

            # Trenowanie modelu na zbiorze treningowym
            model.fit(X_train, y_train)

            # Ocena modelu na zbiorze testowym
            score = model.score(X_test, y_test)

            # Wyświetlenie wyniku dla każdej iteracji
            print("Wynik dla foldu:", score)

    def run(self):
        self.read_config()
        self.model_selection()
        self.cross_validation()
