from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd
from src.utils import get_ex_df
## Here is model training pipeline : 
# When Someone click on the train button this will work !!

def run_training():
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_ingestion()
    ex_df_columns = get_ex_df(train_path=train_data_path)
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_report,best_model_name,best_accuracy = ModelTrainer().initiate_model(train_arr,test_arr)
    return (model_report,best_model_name,best_accuracy,ex_df_columns)

# def get_array_columns():
    