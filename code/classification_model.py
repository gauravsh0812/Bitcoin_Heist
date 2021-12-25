# -*- coding: utf-8 -*-
# import modules
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse

# Argument Parser - take input parameters

parser = argparse.ArgumentParser(description='Classification model')
parser.add_argument('-data', '--data_path', type=str, metavar='', required=True, help='Source path to data file')
parser.add_argument('-label', '--label_path', type=str, metavar='', required=True, help='Source path to label file')
parser.add_argument('-test', '--test_flag',type=int, metavar='', required=True, help='run to test or to train')
parser.add_argument('-train_from_scratch', '--train_from_scratch',type=int, metavar='', required=True, help='if train, should train from scratch')
args = parser.parse_args()


def hyperparameter_tuning():
    
    """
    Tuning the final model with best accuracy
    :param:
    :return: best parameters for the XGBoost Model
    """
    
    print(' ============= Hyper Parameter Tuning using GridSearch Algorithm =============')
    
    # importing valditaion dataset 
    val_X = pd.read_csv('../data/classification/val_data.csv') 
    val_y = pd.read_csv('../data/classification/val_label.csv')

    
    '''
    parameters going to tune:
        eta: learning rate    
        max_depth: higher the depth, more specific relations will be considered by the model\
                   hence increases the chances of overfitting
        subsample: fraction of observation sampled from each tree. Should be small 0.5 to 1
        colsample_bytree: fraction of columns to be sampled randomly from the tree,
        gamma: minimum loss reduction to make tree split, 
        reg_alpha: regularization parameter   
        
    '''
    
    """
    Let's split the entire process into five steps to reduce the tuninig time.
    """
    def sample_tune(LR, gamma, max_depth, subsample, colsample_bytree):
        
        """
        Tuning eta or learning rate
        """
    
        parameters = [
                      {'eta': LR,
                      'gamma': gamma,
                      'max_depth': max_depth,
                      'subsample': subsample,
                      'colsample_bytree': colsample_bytree,
                      'objective':['multi:softmax'],
                      'booster':['gbtree'],
                      'seed':[27]
                        }
                      ]
        
        XGBClf = (XGBClassifier(eta=0.1,  objective='multi:softmax', booster='gbtree', seed = 27, use_label_encoder=False,eval_metric='mlogloss'))
        
        GS = GridSearchCV(estimator = XGBClf, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1 )
        GS.fit(val_X, val_y)
        
        print(GS.cv_results_)
        print(GS.best_params_)
        print(GS.best_score_)
        
        return (GS.best_params_)
    
    # ======= fix the high LR ~0.1 and tune other parameters ==========#
   
    # ======= STEP 1: gamma ==========#
    LR = [0.1]
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_depth = [6]
    subsample = [1]
    colsample_bytree = [1]
    best_features = sample_tune(LR, gamma, max_depth, subsample, colsample_bytree)
    best_gamma = best_features['gamma']
    
    # ======= STEP 2: max_depth ==========#
    LR = [0.1]
    gamma = [best_gamma]
    max_depth = [3,5,7,10]
    subsample = [1]
    colsample_bytree = [1]
    best_features = sample_tune(LR, gamma, max_depth, subsample, colsample_bytree)
    best_max_depth = best_features['max_depth']
        
    # ======= STEP 3: subsample  ==========#
    LR = [0.1]
    gamma = [best_gamma]
    max_depth = [best_max_depth]
    subsample = [0.5, 0.7, 1]
    colsample_bytree = [1]
    best_features = sample_tune(LR, gamma, max_depth, subsample, colsample_bytree)
    best_subsample = best_features['subsample']
    
    # ======= STEP 4: colsample_bytree ==========#
    LR = [0.1]
    gamma = [best_gamma]
    max_depth = [best_max_depth]
    subsample = [best_subsample]
    colsample_bytree = [0.5, 0.7, 1]
    best_features = sample_tune(LR, gamma, max_depth, subsample, colsample_bytree)
    best_colsample = best_features['colsample_bytree']
    
    # ======= STEP 5: LR ==========#
    LR = [0.01, 0.05, 0.1, 1]
    gamma = [best_gamma]
    max_depth = [best_max_depth]
    subsample = [best_subsample]
    colsample_bytree = [best_colsample]
    best_LR = sample_tune(LR, gamma, max_depth, subsample, colsample_bytree)
    best_LR = best_features['eta']
    
    print(' ================= Best Parameters for XGBoost Classifier: =================')
    print(best_features)
    
    return (best_features)


    
def classification(X,y, test=0,  full_train=0):
    """
    classfication model
    :param X: traianing/testing data
    :param y: training/testing label
    :param full_train: flag to enable the full training process
    :return:
    """
    
    def full_training(X,y):
        # Using pipelines to try multiple classifier to see which one gives us the best results
    
        """
        Trying various clssifiers based on the study that has been done in EDA section
        (refer to EDA .ipynb file)
        As the dataset has categorical label to classify, the best way top deal with them is to use 
        either bagging or boosting classifier. Random Forest doesn't even require to encode the catg variable
        We will be using four kind of classifiers:
            Bagging -- RandomForest Classifier
            Boosting -- XGBoost
            K-NN classifier -- KNeighborsClassifier
            SVC -- Support Vector Classifier
        """
    
        # Build Pipelines
        print('Buiding pipelines...')
        print(' ')
        pipeline_feat_SVC = Pipeline([('SVC', SVC(C = 1, kernel = 'rbf', gamma =0.1 , random_state = 27, cache_size = 200))])
        pipeline_feat_Knn = Pipeline([('KNN-Classifier', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))])
        pipeline_feat_RF= Pipeline([('RF-Classifier', RandomForestClassifier(n_estimators = 1, criterion = 'entropy', random_state = 27))])
        pipeline_feat_XGB = Pipeline([('XGOOST-Classifier', XGBClassifier(objective='multi:softmax', booster='gbtree', seed = 27))])
        
        # Fitting the pipeline
        print('Fitting pipelines...')
        print(' ')
        pipe_feat_list = [pipeline_feat_SVC, pipeline_feat_Knn,  pipeline_feat_RF, pipeline_feat_XGB]
        
        score_table=[]
        for pipe in pipe_feat_list:
            print(f'Training {pipe}')
            scores = cross_val_score(pipe, X, y, cv=5)
            score_table.append(scores.mean())
                
    
        # prininting out the final score table
        col_1 = ['SVC', 'K-NN',   'RandomForest', 'XGBoost']
        col_2 = score_table
        Final_table = pd.concat([pd.DataFrame(col_1), pd.DataFrame(col_2)], axis = 1, ignore_index = True)
        Final_table.columns = ['ML_Classifier' ,'Accuracy'] 
        print(' ')
        print('The accuracy table')
        print(Final_table)
        print(' ')
        print(' ')
        print('The maximum accuracy has been achieved by XGBoost Model. Hence we will now tune hyperparameters \
              of the XGBoost model using validation dataset. ')
        
        best_features = hyperparameter_tuning()
        
        return best_features
        
    def best_model_train(X,y, test=0):
        
        """
        Based on the full extent training and tuning results,
        runnnig the best model i.e. XGBoost classifier  using the best parameters.
        
        :param:
        :return: scores
        """
        XGB_clf = XGBClassifier(booster='gbtree', 
                                colsample_bytree= 1, 
                                eta= 0.1, 
                                gamma= 0.5, 
                                max_depth= 10, 
                                objective= 'multi:softmax', 
                                seed= 27, 
                                subsample= 0.7)
        XGB_clf.fit(X,y)

        if test==0:
            scores = cross_val_score(XGB_clf, X, y, cv=5)
            print('training accuracy score:  ',scores.mean())
                    
        else:
            y_pred = XGB_clf.predict(X)
            cm = confusion_matrix(y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            print('confusion matrix:')
            print(cm)
            print(' ')
            print('accuracy score:  ', accuracy)

       
    if full_train==1: 
       print(full_train)
       print(' ============= training from scratch =============')
       full_training(X,y) 
    else:
       best_model_train(X,y, test=test)
       

if __name__ == '__main__':
    
    X = pd.read_csv(args.data_path)
    y = pd.read_csv(args.label_path)
    classification(X,y, test = args.test_flag, full_train=args.train_from_scratch)
    
