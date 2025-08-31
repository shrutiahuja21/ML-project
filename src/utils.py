import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models.
    Returns a dictionary with model names and their R2 test scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                "train_score": train_model_score,
                "test_score": test_model_score
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
