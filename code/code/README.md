# Bitcoin Heist Problem

This model consists of two sub-models: classification model to predict the "team name" responsible for the heist, and a regression model to predict the "income" that has been generated from ransom in the form of Bitcoin Satoshis.

## Requirements
python >3.7

Pandas

NumPy

sklearn

imblearn

SciPy

XGBoost

## Data
A sample of dataset is provided that consists of approximate 60K datapoints. Full dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset

## Model 
Before running models, it is highly recommended to read **"Exploratory Data Analysis(EDA)"** report. It can be found at **"../code/EDA_Bitcoin_Heist.ipynb"**

### Classification Model
To run data preprocessing for classification model
```
python code/data_preprocessing -data 'data/BitcoinHeistData.csv' -clf 1 -reg 0
```
To train classification model for the already hypertuned best performance model 
```
python code/classification_model.py -data 'data/classification/training_data.csv' -label 'data/classification/training_label.csv' -test 0 -train_from_scratch 0
```
Set the `-train_from_scartch 1` to train the classification model from scratch which includes training on all five different classifiers along with the hyperparameter tuning for the best performing model.

To test the classification model
```
python code/classification_model.py -data 'data/classification/testing_data.csv' -label 'data/classification/testing_label.csv' -test 1 -train_from_scratch 0
```

### Regression Model
To run data preprocessing for regression model
```
python code/data_preprocessing -data 'data/BitcoinHeistData.csv' -clf 0 -reg 1
```
To train regression model 
```
python code/regression_model.py -data 'data/regression/training_data.csv' -label 'data/regression/training_label.csv' -test 0 
```
To test the regression model
```
python code/regression_model.py -data 'data/regression/testing_data.csv' -label 'data/regression/testing_label.csv' -test 1 
```
