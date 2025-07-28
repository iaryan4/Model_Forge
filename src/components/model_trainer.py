import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor,CatBoostClassifier

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from xgboost import XGBRegressor ,XGBClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
from src.utils import evaluate_models,get_problem_type
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    ## For Regressor 
    def initiate_model_trainer_regressor(self,train_array,test_array):

        X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

        models_regressor = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Logistic Regressor" : LogisticRegression(),
                "SVR" : SVR(),
            }
        params_regressor={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regressor":{
                    'penalty' : ['l1', 'l2', 'elasticnet', None],
                    'C': [100, 10, 1.0, 0.1, 0.01],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag','saga'],
                },
                "SVR" : {
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'C': [0.1, 1, 10, 100, 1000],
                    'kernel': ['rbf']
                },
            }
        return models_regressor,params_regressor

    ## For Classification
    def initiate_model_trainer_classifier(self, train_array, test_array):

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models_classifier = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(),
            "CatBoost Classifier": CatBoostClassifier(verbose=False),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "SVC": SVC(),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier()
        }

        params_classifier = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2', None],
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2'],
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': ['sqrt', 'log2']
            },
            "Logistic Regression": {
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'C': [100, 10, 1.0, 0.1, 0.01],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            },
            "XGBClassifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            "CatBoost Classifier": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [50, 100]
            },
            "AdaBoost Classifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            "SVC": {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            "Naive Bayes": {},  # GaussianNB has no hyperparameters to tune
            "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        }
        
        return models_classifier,params_classifier

    def initiate_model(self,train_array,test_array):
        models_reg,params_reg = self.initiate_model_trainer_regressor(train_array, test_array)
        models_cls,params_cls = self.initiate_model_trainer_classifier(train_array, test_array)
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        problem__type = get_problem_type()
        if problem__type == 'Regression':
            model_report,best_model,best_model_name,best_acc = evaluate_models(X_train,X_test,y_train,y_test,models=models_reg,params=params_reg)
        elif problem__type == 'Classification':
            model_report,best_model,best_model_name,best_acc = evaluate_models(X_train,X_test,y_train,y_test,models=models_cls,params=params_cls)
        save_object(ModelTrainerConfig.trained_model_file_path,best_model)
        print(model_report)
        return model_report,best_model_name,best_acc

        ## Saving the model

