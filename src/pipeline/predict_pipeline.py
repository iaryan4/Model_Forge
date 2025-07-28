import os
import pandas as pd
import numpy as np
from src.components.data_transformation import DataAfterEDA,drop_name_like_columns
from src.utils import load_model,load_preprocessor
from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
def get_columns(df):
    target_col = DataAfterEDA().get_target()
    new_df = df.drop([target_col, 'Unnamed: 0'], axis=1, errors='ignore')
    columns=new_df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in new_df.dtypes.items()}
    print(columns)
    return columns,dtypes


def initiate_training():
    target = DataAfterEDA().get_target()
    train_df,_ = DataAfterEDA().get_df_info()
    tar_dtype = train_df[target].dtype
    df = pd.read_csv(os.path.join('artifact','user_test.csv'))
    df.drop(columns=[target], errors='ignore', inplace=True)
    df_new = drop_name_like_columns(df)
    preprocessor_obj = load_preprocessor(os.path.join('artifact','preprocessor.pkl'))
    model_obj = load_model(os.path.join('artifact','model.pkl'))
    X = df_new.drop(columns=[target],errors='ignore')
    X_scaled = preprocessor_obj.transform(X)
    y_pred = model_obj.predict(X_scaled)
    # Convert predictions to a DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=[target])
    y_pred_df = y_pred_df.astype(tar_dtype)

    # Concatenate predictions with the original DataFrame
    final_df = pd.concat([df.reset_index(drop=True), y_pred_df], axis=1)
    return final_df

    
if __name__ == '__main__':
    get_columns()