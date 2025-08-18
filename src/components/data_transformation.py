import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import os
import pickle


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Categorical columns encoding completed")
            logging.info("Numerical columns scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Debug info
            print("Train path:", train_path)
            print("Test path:", test_path)
            print("Does train exist?", os.path.exists(train_path))
            print("Does test exist?", os.path.exists(test_path))

            # 1. Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            target_column = "math score"
            numerical_columns = ['writing score', 'reading score']

            # 2. Split input features and target
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Splitting into features and target completed")

            # 3. Get preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            # 4. Fit & Transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            # 5. Concatenate features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 6. Save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessing_obj, f)

            logging.info("Preprocessor pickle file saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        

def find_file(folder, filename):
    """Find a file in folder ignoring case."""
    for f in os.listdir(folder):
        if f.lower() == filename.lower():
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{filename} not found in {folder}")

if __name__ == "__main__":
    data_folder = os.path.join(PROJECT_ROOT, "dataa")
    file_path = os.path.join(data_folder, "StudentsPerformance.csv")

    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save temporary train/test CSVs
    train_path = os.path.join(data_folder, "Train.csv")
    test_path = os.path.join(data_folder, "Test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(
       train_path=train_path,
       test_path=test_path
    )

    print("Train Array Shape:", train_arr.shape)
    print("Test Array Shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)