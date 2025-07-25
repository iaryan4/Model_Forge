## We'll will perform EDA and the transformation part like encoding or scaling here :
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.utils import save_object

# NAME REMOVAL
def drop_name_like_columns(df):
    import re

    # Define common patterns related to names and identifiers
    name_like_keywords = [
        "name", "person", "customer", "client", "user", "username",
        "firstname", "lastname", "full_name", "fullname", "contact_name",
        "passenger", "owner", "author"
    ]

    cols_to_drop = []

    for col in df.columns:
        col_lower = col.lower()
        # Match column names that include any of the name-like keywords
        if any(re.search(rf"\b{k}\b", col_lower) for k in name_like_keywords):
            cols_to_drop.append(col)
        # OR: if high-cardinality object column, assume identifier-like
        elif df[col].dtype == 'object' and df[col].nunique() > 100:
            cols_to_drop.append(col)

    # Drop safely
    return df.drop(columns=cols_to_drop, errors='ignore')




##if NaN values are too much like more than 50%

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"proprocessor.pkl")

class DataAfterEDA:
    def __init__(self):
        pass
    def get_target(self):
        target = ''
        with open (os.path.join('artifact','target.txt'),'r') as fp :
            target = fp.read()
        return target
    def get_df_info(self):
                # EDA

        df_train = pd.read_csv(os.path.join('artifact','train.csv'))
        df_test = pd.read_csv(os.path.join('artifact','test.csv'))
        
        # Drop name-like columns by default
        df_train = drop_name_like_columns(df_train)
        df_test = drop_name_like_columns(df_test)

        return df_train,df_test


    def update_data_after_EDA(self):
        df_train,df_test = self.get_df_info()
        df_train.to_csv(os.path.join('artifact','train.csv'))
        df_test.to_csv(os.path.join('artifact','test.csv'))
        
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        df_train,df_test = DataAfterEDA().get_df_info()
        
        target = DataAfterEDA().get_target()

        # Numerical columns & Categorical Columns :
        numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if target in numerical_columns:
            numerical_columns.remove(target)
        elif target in categorical_columns:
            numerical_columns.remove(target)
        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())

            ]
        )

        cat_pipeline=Pipeline(

            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]

        )

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

            ]


        )

        return preprocessor

    def initiate_data_transformation(self,train_path,test_path):

            target = DataAfterEDA().get_target()
            DataAfterEDA().update_data_after_EDA()
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            print("Train Data Columns:", train_df.columns.tolist())
            print("Test Data Columns:", test_df.columns.tolist())
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= target
            # print(target_column_name)

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            # print(target_feature_train_df)

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            # print(input_feature_test_df.columns)


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
    

if __name__ == '__main__':
    pass