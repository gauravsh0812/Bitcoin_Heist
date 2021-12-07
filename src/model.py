# -*- coding: utf-8 -*-

def classification(train_X, train_y, val_X, val_y):
    
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # converting to Dataframe
    train_X = pd.DataFrame(train_X)
    train_y = pd.DataFrame(train_y)
    
    # Using pipelines to try multiple classifier to see which one gives us the best results
    def pipelining_clssifiers():
        """
        Trying various clssifiers to 
        """
        pipeline_feat_DT= Pipeline([('classifier_feat_DT', DecisionTreeClassifier( criterion = 'entropy', random_state = 0))])
        pipeline_feat_RF= Pipeline([('classifier_feat_RF', RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0))])
        pipeline_feat_LT = Pipeline([('classifier_feat_LT', LogisticRegression(random_state = 0))])
        pipeline_feat_SVC = Pipeline([('classifier_feat_SVC', SVC(C = 1, kernel = 'rbf', gamma =0.1 , random_state = 0, cache_size = 200))])
        pipeline_feat_Knn = Pipeline([('classifier_feat_Knn', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))])
        pipeline_feat_Naive_Bayes = Pipeline([('classifier_feat_Naive_Bayes', GaussianNB())])

    
    