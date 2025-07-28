import os 
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore")

#Let's create a paht for the artifact file and where to save the files:
@dataclass # this automatically generate the __init__() and some other class things
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")

class DataIngestion:
        def __init__(self):
              self.ingestion_config = DataIngestionConfig()
        def initiate_ingestion(self):
              df = pd.read_csv('artifact/raw_data.csv')
              train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)
              
              train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

              test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
              return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_ingestion()
    print(train_data,test_data) # = artifact\train.csv
    data_transformation=DataTransformation()
    # data_transformation.initiate_data_transformation(train_data,test_data)
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_report = ModelTrainer().initiate_model(train_arr,test_arr)
