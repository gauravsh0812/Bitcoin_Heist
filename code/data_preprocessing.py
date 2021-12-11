import numpy as np
import pandas as pd
import argparse

# Argument Parser - take input parameters

parser = argparse.ArgumentParser(description='Classification model')
parser.add_argument('-data', '--data_path', type=str, metavar='', required=True, help='Source path to data file')
parser.add_argument('-clf', '--classification',type=int, metavar='', required=True, help='preprocessing for classification model')
parser.add_argument('-reg', '--regression',type=int, metavar='', required=True, help='preprocessing for regression model')
args = parser.parse_args()


def import_data(data_path):
    """
    Import dataset 
    :param data_path:
    :return: dataframe
    """
    df = pd.read_csv(data_path)
    print('importing dataset...')
    print(df.head())
    return(df)
   

def missing_values(df):  
    """
    checking missing values
    :param df: dataframe
    :return: dataframe with zero missing values
    """
    print('dealing with missing values...')
    df_missingValues = df.isnull().sum()
    if sum(df_missingValues) == 0:
        print('There are no missing values in the dataset. For further detail, \
              please refer to EDA ipynb file.')
    else:
        # replace np.NAN values with median 
        df.apply(lambda x: x.fillna(x.median()),axis=0)
    
    return df

def drop_features(df, clf, reg):
    """
    dropping less important or irrelevant features and rows
    :param df:
    :return: df without the given features and rows
    """
    print('*************'*5) 
    print('For explanation and plots from now onwrads, refer to EDA ipynb file')
    print('*************'*5)
    
    if clf ==1:
        for col in ['address', 'day', 'neighbors', 'looped']:
            df.drop(col, axis=1, inplace=True)
        index_2018 = df[df['year']==2018].index
        df.drop(index_2018, inplace = True)

        # Dropping "label" that has less than 100 datapoints
        label_dict = df['label'].value_counts().to_dict()
        print(' ')
        print('dropping labels with less than 100 datapoints/frequency')
        for k,v in label_dict.items():
            if v<100:
                index_k = df[df['label']==k].index
                df.drop(index_k, inplace = True)

    if reg ==1:
        for col in ['address', 'day', 'neighbors', 'looped', 'year', 'label']:
            df.drop(col, axis=1, inplace=True)
            
    print('Dropping irrelevant features and rows...')
    '''
    # Dropping "label" that has less than 100 datapoints
    label_dict = df['label'].value_counts().to_dict()
    print(' ')
    print('dropping labels with less than 100 datapoints/frequency')
    for k,v in label_dict.items():
        if v<100: 
            index_k = df[df['label']==k].index
            df.drop(index_k, inplace = True)
    '''
    print(' ')
    #print('sample dataset at this stage looks like...')
    #print(df.head())
    return df

def dealing_with_outliers(df):
    """
    dealing with outliers 
    :param df:
    :return: df having outlier replaced with the boundary values    
    """
    # take negative log  of feature 'weight' to better visualize it
    df['weight'] = -np.log(df['weight'])
    print('taking negative log of the weight feature...')
    print(' ')

    print('dealing with outliers... ') 
    print(' ')

    # detectig outliers 
    def detect_outliers_iqr(data):
        outliers = []
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
    print('encoding categorical features i.e. year and label...')
    data_enc = pd.get_dummies(data=df['year'],drop_first=True)
    df = pd.concat([pd.DataFrame(data_enc), df.iloc[:,1:]], axis=1)
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    #print('sample dataset at this stage...')
    #print(df.head())
    print(' ')
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
    print('dealing with imbalanced dataset...')
    print('  ')

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
    """
    shuffling thae dataset
    :param df:
    :return df:
    """
    print('shuffling the dataset...')
    from sklearn.utils import shuffle
    return shuffle(df)


def separate_dataset(df):
    """
    separating datset into X:features, y:final_label
    :param df:
    :param final_col: column/feature that will be serves as the final label
    :return: X:features, y:final_label for classification or regression
    """    
    print('separating features(X) and final label(y)... ')

    y = df.iloc[:,-1]
    X = df.iloc[:, :-1]
    
    return (X,y)



def splitting_dataset(df_to_split, test_split=False):
    """
    separating dataset into features and final label.
    also, separting dataset into training and testing dataset.
    :param df:
    :return: train_X, train_y, test_X, test_y
    """
    if test_split:
        print('dividing  the dataset into test and train...')
        print(' ')
        N = int(0.9*len(df_to_split))
        train_df, test_df = df_to_split[:N], df_to_split[N:]
        
        return (train_df, test_df)
    
    else:    
        # splitting the train_df into train, val features and labels
        train_X, train_y = separate_dataset(df_to_split)
        print('splitting the dataset into train and validation datasets...')
        print(' ')
        from sklearn.model_selection import train_test_split
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.1, random_state = 0)
        
        return (train_X, train_y, val_X, val_y)


def scaling(datasets, COL, clf, reg):
    """
    standardize the dataset using StandardScaler
    fitting and transforming training and validation datset
    while transforming testing dataset using the 
    :param df:
    :return df:
    """
    print('scaling the datasets...')
    print(' ')
    from sklearn.preprocessing import StandardScaler
    if clf ==1:
        train_X, test_X, val_X = datasets
        sc_x = StandardScaler()
        for idx, x in enumerate([train_X, val_X]):
            x.reset_index(drop=True, inplace=True)
            x_years = x.iloc[:, :6]
            tobe_scaled = x.iloc[:, 6:]
            x_scaled = sc_x.fit_transform(tobe_scaled)
            x_scaled = pd.DataFrame(x_scaled)
            x_scaled.columns = COL[6:-1]
            final_scaled_X = pd.concat([x_years, x_scaled], axis=1)
            
            if idx==0: 
                train_X = final_scaled_X
            else: 
                val_X = final_scaled_X
            
        test_X.reset_index(drop=True, inplace=True)
        x_scaled_test = sc_x.transform(test_X.iloc[:, 6:])
        x_scaled_test = pd.DataFrame(x_scaled_test)
        x_scaled_test.columns = COL[6:-1]
        test_X = pd.concat([pd.DataFrame(test_X.iloc[:, :6]), x_scaled_test], axis=1)
        
        return (train_X, test_X, val_X)
    
    elif reg==1:
        train_X, test_X, val_X, train_y, val_y, test_y = datasets
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        LIST=[train_X, val_X, train_y, val_y]
        for idx, x in enumerate(LIST):
            x.reset_index(drop=True, inplace=True)
            x=LIST[idx]
            if idx<2:
                x_scaled = pd.DataFrame(sc_x.fit_transform(x.iloc[:, 0:]))
                x_scaled.columns = COL[0:-1]
            else:
                y_scaled = pd.DataFrame(sc_y.fit_transform(x.values.reshape(-1,1)))
                y_scaled.columns = ['income']            
            
            if idx==0: 
                train_X = x_scaled
            elif idx==1: 
                val_X = x_scaled
            elif idx==2: 
                train_y = y_scaled
            else: 
                val_y = y_scaled
            
        test_X.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)
        x_scaled_test = pd.DataFrame(sc_x.transform(test_X.iloc[:, 0:]))
        x_scaled_test.columns = COL[0:-1]
        y_scaled_test = pd.DataFrame(sc_y.transform(test_y.values.reshape(-1,1)))
        y_scaled_test.columns = ['income']
        
        return (train_X, test_X, val_X, train_y, val_y, test_y)
    
        
def saving_dataset(train_X, test_X, train_y, test_y, val_X, val_y, clf, reg):
    """
    Finally, save the hard work we have done 
    :param: final datasets
    """
    import os
    for PATH in ['data/classification', 'data/regression']:
        if not os.path.exists(PATH):
            os.mkdir(PATH)
    
    if clf==1: folder='classification'
    elif reg==1: folder= 'regression' 
    
    print('saving the datasets...')
    train_X.to_csv(f'data/{folder}/training_data.csv', index=False)
    test_X.to_csv(f'data/{folder}/testing_data.csv', index=False)
    val_X.to_csv(f'data/{folder}/val_data.csv', index=False)
    train_y.to_csv(f'data/{folder}/training_label.csv', index=False)
    test_y.to_csv(f'data/{folder}/testing_label.csv', index=False)
    val_y.to_csv(f'data/{folder}/val_label.csv', index=False)
    


def preprocessing(data_path, clf, reg):
    # Importing Dataset 
    df = import_data(data_path)
    # dealing with missing values
    df = missing_values(df)
    # dropping irrelevant features
    df = drop_features(df, clf, reg)
    # dealing with outliers and skewness
    df = dealing_with_outliers(df)
    # shuffle the datasets 
    df = shuffle_dataset(df)  
    # splitting the dataset into train, test as testing dataset shouldn't be augmented
    train_df, test_df = splitting_dataset(df, test_split=True)
    # dealing with imbalance datset i.e. label
    if clf ==1:
        train_df = imbalance_dataset(train_df)
    # dealing with categorical data
    if clf ==1:
        train_df = categorical_feature_converstion(train_df)   
        test_df = categorical_feature_converstion(test_df)   
    # splitting the dataset into train, Validation
    train_X, train_y, val_X, val_y = splitting_dataset(train_df, test_split=False)
    # splitting the test dataset into features and label
    test_X, test_y = separate_dataset(test_df)
    # standardization
    if clf==1:
        train_X, test_X, val_X = scaling([train_X, test_X, val_X], train_df.columns, clf, reg) 
    if reg==1:
        train_X, test_X, val_X, train_y, test_y, val_y = scaling([train_X, test_X, val_X, train_y, test_y, val_y], train_df.columns, clf, reg)
    # Saving final training and testing datasets
    saving_dataset(train_X, test_X, train_y, test_y, val_X, val_y, clf, reg)
    

if __name__ == "__main__":
    preprocessing(args.data_path, args.classification, args.regression)
    
