
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import argparser


# Paths
data_path = '../data/BitcoinHeistData.csv'

def import_data(data_path):
    """
    Import dataset 
    :param data_path:
    :return: dataframe
    """
    return(pd.read_csv(data_path))

def missing_values(df):  
    """
    checking missing values
    :param df: dataframe
    :return: dataframe with zero missing values
    """
    
    df_missingValues = df.isnull().sum()
    if sum(df_missingValues) == 0:
        print('There are no missing values in the dataset. For further detail, \
              please refer to EDA ipynb file.')
    else:
        # replace np.NAN values with median 
        df.apply(lambda x: x.fillna(x.median()),axis=0)
    
    return df

def drop_features(df):
    """
    dropping less important or irrelevant features
    :param df:
    :return: df without the given features
    """
    
    for col in ['address', 'day', 'neighbors', 'looped']:
        df.drop(col, axis=1, inplace=True)
    
    return df

def dealing_with_outliers(df):
    """
    dealing with outliers 
    :param df:
    :return: df having outlier replaced with the boundary values    
    """
    # detectig outliers 
    outliers = []
    def detect_outliers_iqr(data):
        data = sorted(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        # print(q1, q3)
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        # print(lwr_bound, upr_bound)
        for i in data: 
            if (i<lwr_bound or i>upr_bound):
                outliers.append(i)
        return outliers, lwr_bound, upr_bound
    
    for feature in ['length', 'weight', 'count',  'income']:
        
        # detecting outliers
        feature_outliers, lwr_bound, upr_bound = detect_outliers_iqr(df[feature])
        
        # Replacing outliers
        df[feature] = np.where(df[feature]<lwr_bound, lwr_bound, df[feature])
        df[feature] = np.where(df[feature]>upr_bound, upr_bound, df[feature])
          
    return df   

def categorical_feature_converstion(df):
    
    return df

def normalization(df):
    
    return df

def preprocessing():
    # Importing Dataset 
    df = import_data(data_path)
    # dealing with missing values
    df = missing_values(df)
    # dropping irrelevant features
    df = drop_features(df)
    # dealing with outliers and skewness
    df = dealing_with_outliers(df)
    # dealing with categorical data
    df = categorical_feature_converstion(df)
    # normalization 
    df = normalization(df)


if __name__ == "__main__":
    preprocessing()
    