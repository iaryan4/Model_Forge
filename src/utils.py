import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold,StratifiedKFold
import matplotlib.pyplot as plt
import streamlit as st

def save_object(file_path, obj):
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, r2_score
    
    all_reports = []
    all_best_models = []
    all_best_params = []
    
    types = get_problem_type()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if types == 'Classification' else KFold(n_splits=3, shuffle=True, random_state=42)
    
    n_iters = get_n_iters(X_train)
    
    for _ in range(3):  # Run multiple times to explore randomness
        report = {}
        for model_name in models.keys():
            model = models[model_name]
            param_grid = params.get(model_name, {})
            grid_search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                cv=cv,
                scoring='accuracy' if types == 'Classification' else 'r2',
                n_jobs=-1,
                n_iter=n_iters
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_pred = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred) if types == 'Classification' else r2_score(y_test, y_pred)

            report[model_name] = acc_score
            all_best_models.append((model_name, best_model, best_params, acc_score))
        
        all_reports.append(report)

    # Find the best overall
    best_score = -1
    best_model_name = None
    best_model = None
    b_score = -1
    
    for model_name, model_obj, params, score in all_best_models:
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model_obj
    best_report = {}
    for r in all_reports:
        if r.get(best_model_name) == best_score:
            best_report = r
            break  # Found the matching report, no need to continue
    report = best_report

    return (
        report,
        best_model,
        best_model_name,
        best_score
    )


def get_problem_type():
    with open(os.path.join('artifact','target.txt')) as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]
    return lines[1]

def get_n_iters(X):
        # Decide n_iter based on data size
    data_size = len(X)
    if data_size <= 1000:
        n_iter = 5
    elif data_size <= 10000:
        n_iter = 10
    elif data_size <= 100000:
        n_iter = 20
    else:
        n_iter = 30
    return n_iter


def get_ex_df(train_path):
    df=pd.read_csv(train_path)
    return df

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    return model

def load_preprocessor(preprocessor_path):
    with open(preprocessor_path, 'rb') as f:
        model = dill.load(f)
    return model