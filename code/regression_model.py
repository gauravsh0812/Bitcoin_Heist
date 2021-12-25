# -*- coding: utf-8 -*-
# import modules
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import argparse

# Argument Parser - take input parameters

parser = argparse.ArgumentParser(description='Classification model')
parser.add_argument('-data', '--data_path', type=str, metavar='', required=True, help='Source path to data file')
parser.add_argument('-label', '--label_path', type=str, metavar='', required=True, help='Source path to label file')
parser.add_argument('-test', '--test_flag',type=int, metavar='', required=True, help='run to test or to train')
args = parser.parse_args()


    
def regression(X,y, test=0):
    """
    regression model
    :param X: traianing/testing data
    :param y: training/testing label
    :param full_train: flag to enable the full training process
    :return:
    """
    
    def full_training(X,y):
        # Using pipelines to try multiple classifier to see which one gives us the best results
    
        """
        Trying various regressors based on the study that has been done in EDA section
        (refer to EDA .ipynb file)
        As the dataset has categorical label to classify, the best way top deal with them is to use 
        either bagging or boosting regressor. Random Forest doesn't even require to encode the catg variable
        We will be using four kind of regressors:
            Bagging -- RandomForest regressor
            Boosting -- XGBoost
            K-NN regressor -- regressor
            SVR -- Support Vector regressor
        """
    
        # Build Pipelines
        print('Buiding pipelines...')
        print(' ')
        pipeline_feat_SVR = Pipeline([('SVR', SVR(C = 1, kernel = 'rbf', gamma =0.1 , cache_size = 200))])
        pipeline_feat_Knn = Pipeline([('KNN-Regressor', KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2))])
        pipeline_feat_RF= Pipeline([('RF-Regressor', RandomForestRegressor(n_estimators = 1, criterion = 'squared_error', random_state = 27))])
        pipeline_feat_XGB = Pipeline([('XGBOOST-Regressor', XGBRegressor(objective='reg:squarederror', seed = 27))])
        
        # Fitting the pipeline
        print('Fitting pipelines...')
        print(' ')
        pipe_feat_list = [pipeline_feat_SVR, pipeline_feat_Knn,  pipeline_feat_RF, pipeline_feat_XGB]
        
        score_table=[]
        for pipe in pipe_feat_list:
            print(f'Training {pipe}')
            scores = cross_val_score(pipe, X, y, cv=5)
            score_table.append(scores.mean())
                
    
        # prininting out the final score table
        col_1 = ['SVR', 'K-NN',   'RandomForest', 'XGBoost']
        col_2 = score_table
        Final_table = pd.concat([pd.DataFrame(col_1), pd.DataFrame(col_2)], axis = 1, ignore_index = True)
        Final_table.columns = ['ML_Regressor' ,'Accuracy'] 
        print(' ')
        print('The accuracy table')
        print(Final_table)
    
    full_training(X,y)

if __name__ == '__main__':
    
    X = pd.read_csv(args.data_path)
    y = pd.read_csv(args.label_path)
    regression(X,y, test = args.test_flag)
    
