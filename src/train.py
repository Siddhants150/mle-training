import pandas as pd
import pickle
import os
import argparse
import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import date

today = date.today()

d1 = today.strftime("%d/%m/%Y")

modelFolder = os.path.join("artifacts", "models")
inputFolder = os.path.join("datasets", "data")

logPath = os.path.join("logs", d1)
logLevel = logging.INFO
consoleLog = 1


def trainLinearModel(X, Y, fileName):
    logger.debug("Entered train linear regression function")
    linearRegression = LinearRegression()
    linearRegression.fit(X, Y)
    YPred = linearRegression.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    logger.info("Linear Regression RMSE on Training: ")
    logger.info(rmse)
    pickle.dump(
        linearRegression,
        open(os.path.join(modelFolder, fileName), "wb"),
    )


def trainTreeRegressor(X, Y, fileName):
    logger.debug("Entered train tree regression function")
    DTR = DecisionTreeRegressor(random_state=42)
    DTR.fit(X, Y)
    YPred = DTR.predict(X)
    mse = mean_squared_error(Y, YPred)
    rmse = np.sqrt(mse)
    logger.info("Decision Tree Regressor Regression RMSE on Training: ")
    logger.info(rmse)
    pickle.dump(DTR, open(os.path.join(modelFolder, fileName), "wb"))


def trainRandomForestRegressorRandomSearch(X, Y, fileName):
    logger.debug("Entered train random regression with random search function")
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
    logger.info("Random Forest Randomized Search RMSE on Training: ")
    logger.info(rmse)
    pickle.dump(model, open(os.path.join(modelFolder, fileName), "wb"))


def trainRandomForestRegressorGridSearch(X, Y, fileName):
    logger.debug(
        """Entered train random forest
           regression with grid search function"""
    )
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    RFR = RandomForestRegressor(random_state=42)
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
    logger.info("Random Forest Regressor Grid Search RMSE on Training: ")
    logger.info(rmse)
    pickle.dump(model, open(os.path.join(modelFolder, fileName), "wb"))


try:
    argumentParser = argparse.ArgumentParser(
        prog="train",
        description="training the data",
    )
    argumentParser.add_argument(
        "--trainfolder",
        action="store",
        help="mention the train folder name",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--trainfile",
        help="mention the name of train file",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--outputfolder",
        help="mention the name of output folder",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--log_level",
        help="mention the log level",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--log_path",
        help="mention the log path",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--no_console_log",
        help="toggle console log",
        action="store_true",
        required=False,
    )
    arguments = argumentParser.parse_args()
    if arguments.trainfolder:
        inputFolder = os.path.join("datasets", arguments.trainfolder)
    if arguments.outputfolder:
        modelFolder = os.path.join("artifacts", arguments.outputfolder)
    if arguments.log_path:
        logFolder = os.path.join(arguments.log_path, "ingestData.txt")
    print(arguments)

except Exception as e:
    print("error occured")
    print(e)

try:
    if arguments.log_level == "DEBUG":
        logLevel = logging.DEBUG
    elif arguments.log_level == "INFO":
        logLevel = logging.INFO
    elif arguments.log_level == "WARNING":
        logLevel = logging.WARNING
    elif arguments.log_level == "ERROR":
        logLevel = logging.ERROR
    elif arguments.log_level == "CRITICAL":
        logLevel = logging.CRITICAL

    if arguments.log_path:
        logPath = os.path.join(arguments.log_path, "logs")

    if not os.path.isdir(logPath):
        os.makedirs(logPath)

    logFile = os.path.join(logPath, "train.log")

    if arguments.no_console_log:
        consoleLog = 0

    try:
        logging.basicConfig(filename=logFile, level=logLevel)
        logger = logging.getLogger("train")
    except Exception as e:
        print("error occured")
        print(e)

    if consoleLog:
        ch = logging.StreamHandler()
        ch.setLevel(logLevel)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s -%(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

except Exception as e:
    logger.error("Error Occured")
    logger.error(e)

try:
    if arguments.trainfile:
        df = pd.read_csv(
            os.path.join(inputFolder, arguments.trainfile + ".csv"),
        )
    else:
        df = pd.read_csv(os.path.join(inputFolder, "train.csv"))
    print(df.head())
    trainX = df.drop("median_house_value", axis=1)
    trainY = df["median_house_value"].copy()
    print(trainX.head())
    print(trainY.head())
    if not os.path.isdir(modelFolder):
        os.makedirs(modelFolder)
    trainLinearModel(trainX, trainY, "linearRegression")
    trainTreeRegressor(trainX, trainY, "decisionTreeRegressor")
    trainRandomForestRegressorRandomSearch(
        trainX,
        trainY,
        "randomForestRegressorWithRandomSearch",
    )
    trainRandomForestRegressorGridSearch(
        trainX,
        trainY,
        "randomForestRegressorWithGridSearch",
    )

except Exception as e:
    logger.error("error occured")
    logger.error(e)
