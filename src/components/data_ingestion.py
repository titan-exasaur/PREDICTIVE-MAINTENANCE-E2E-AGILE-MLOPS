import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestConfig:
    train_data_path: str = os.path.join("assets", "train.csv")
    test_data_path: str = os.path.join("assets", "test.csv")
    raw_data_path: str = os.path.join("assets", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestConfig()

    def download_dataset(self):
        try:
            logging.info("Starting to download the dataset")
            os.system('kaggle datasets download -d stephanmatzka/predictive-maintenance-dataset-ai4i-2020 -p ./assets')
            os.system('unzip ./assets/predictive-maintenance-dataset-ai4i-2020.zip -d ./assets')
            logging.info("Dataset downloaded and unzipped successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join('assets', 'ai4i2020.csv'))
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for saving data exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            os.remove(os.path.join('assets', 'predictive-maintenance-dataset-ai4i-2020.zip'))
            os.remove(os.path.join('assets', 'ai4i2020.csv'))
            logging.info("Unzipped dataset deleted successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.download_dataset()
    train_data, test_data = obj.initiate_data_ingestion()

    # Uncomment to use data transformation and model training
    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
