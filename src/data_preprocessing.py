
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
    """
    Converting categorical features to numerical 
    using Label Encoder(final  label) or One Hot Encoder(year).
    :param df:
    :return modified df:
    """    
    
    # converting label to numerical values/indices 
    le = LabelEncoder()
    df['label'] = le.fit_transform(features['label'])
    

    dummies = pd.get_dummies(df['year'])
    res = pd.concat([df, dummies], axis=1)
    
    return df

def imbalance_dataset(df):
    """
    dealing with the imbalanced dataset in following steps
    1) cut down the 'white' label as it has ~10x more datapoints        
    2) we will use SMOTE algorithm to create synthetic datapoints
    :param df:
    :return: balanced dataset df
    """
    # randomly dropping datapoints having 'white' label
    tobe_dropped = len(df[df['label']=="white"]) - len(df[df['label']=='paduaCryptoWall'])
    df.drop(df.loc[df['label']=='white'].sample(frac=0.97).index, inplace=True)
    
    # SMOTE algorithm
    # SMOTE to create synthetic datapoints

    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    
    # import library
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE()
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(X, y)
    df = pd.concat([pd.DataFrame(x_smote), pd.DataFrame(y_smote)], axis=1) 
    
    return df

def shuffle_dataset(df):
    from sklearn.utils import shuffle
    return shuffle(df)

def separate_dataset(df, final_col):
    """
    separating datset into X:features, y:final_label
    :param df:
    :param final_col: column/feature that will be serves as the final label
    :return: X:features, y:final_label for classification or regression
    """    
    df_copy = df.copy()
    y = df_copy[final_col]
    X = df_copy.drop(final_col)
    
    return X,y


def splitting_dataset(df, seperate=separate):
    """
    separating dataset into features and final label.
    also, separting dataset into training and testing dataset.
    :param df:
    :return: train_X, train_y, test_X, test_y
    """
    
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 0)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.1, random_state = 0)
    
    return (train_X, test_X, train_y, test_y, val_X, val_y)


def standardisation(df):
    """
    standardize the dataset using StandardScaler
    :param df:
    :return df:
    """
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    
    
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
    # dealing with imbalance datset i.e. years
    df = imbalance_dataset(df)
    # shuffle the dataset 
    df = shuffle_dataset(df)
    # separate_dataset
    X, y = separate_dataset(df, final_col)
    # normalization 
    X = standardisation(X)    
    # splitting thje dataset 
    train_X, test_X, train_y, test_y, val_X, val_y = splitting_dataset(df, seperate=separate)
    

if __name__ == "__main__":
    preprocessing()
    