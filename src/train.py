import pandas as pd
import pickle
import os
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import numpy as np
from sklearn.metrics import mean_squared_error

modelFolder = os.path.join('artifacts', 'models')
inputFolder = os.path.join('datasets', 'data')

def trainLinearModel(X, Y, fileName):
    linearRegression = LinearRegression()
    linearRegression.fit(X, Y)
    YPred = linearRegression.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    print('Linear Regression RMSE on Training: ', rmse)
    pickle.dump(linearRegression, open(os.path.join(modelFolder, fileName), 'wb'))

def trainTreeRegressor(X, Y, fileName):
    DTR = DecisionTreeRegressor(random_state = 42)
    DTR.fit(X, Y)
    YPred = DTR.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    print('Decision Tree Regressor Regression RMSE on Training: ', rmse)
    pickle.dump(DTR, open(os.path.join(modelFolder, fileName), 'wb'))

def trainRandomForestRegressorRandomSearch(X, Y, fileName):
    RFR = RandomForestRegressor()
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    randomSearch = RandomizedSearchCV(
        RFR,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    randomSearch.fit(X, Y)
    model = randomSearch.best_estimator_
    YPred = model.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    print('Random Forest Randomized Search RMSE on Training: ', rmse)
    pickle.dump(model, open(os.path.join(modelFolder, fileName), 'wb'))

def trainRandomForestRegressorGridSearch(X, Y, fileName):
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    RFR = RandomForestRegressor(random_state = 42)
    gridSearch = GridSearchCV(
        RFR,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    gridSearch.fit(X, Y)
    model = gridSearch.best_estimator_
    YPred = model.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    print('Random Forest Regressor Grid Search RMSE on Training: ', rmse)
    pickle.dump(model, open(os.path.join(modelFolder, fileName), 'wb'))

try:
    argumentParser = argparse.ArgumentParser(prog = 'train', description = 'training the data')
    argumentParser.add_argument('--trainfolder', action = 'store', type = str, required = False)
    argumentParser.add_argument('--trainfile', action = 'store', type = str, required = False)
    argumentParser.add_argument('--outputfolder', action = 'store', type = str, required = False)
    arguments = argumentParser.parse_args()
    if arguments.trainfolder:
        inputFolder = os.path.join('datasets', arguments.trainfolder)
    if arguments.outputfolder:
        modelFolder = os.path.join('artifacts', arguments.outputfolder)
    print(arguments)

except Exception as e:
    print('error occured')
    print(e)

try:
    if arguments.trainfile:
        df = pd.read_csv(os.path.join(inputFolder, arguments.trainfile + '.csv'))
    else:
        df = pd.read_csv(os.path.join(inputFolder, 'train.csv'))
    print(df.head())
    trainX = df.drop('median_house_value', axis = 1)
    trainY = df['median_house_value'].copy()
    print(trainX.head())
    print(trainY.head())
    if not os.path.isdir(modelFolder):
            os.makedirs(modelFolder)
    trainLinearModel(trainX, trainY, 'linearRegression')
    trainTreeRegressor(trainX, trainY, 'decisionTreeRegressor')
    trainRandomForestRegressorRandomSearch(trainX, trainY, 'randomForestRegressorWithRandomSearch')
    trainRandomForestRegressorGridSearch(trainX, trainY, 'randomForestRegressorWithGridSearch')

except Exception as e:
    print('error occured')
    print(e)
    
