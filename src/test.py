import pandas as pd
import os
import argparse
import pickle
import logging
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import date

today = date.today()

d1 = today.strftime("%d/%m/%Y")

modelFolder = os.path.join("artifacts", "models")
inputFolder = os.path.join("datasets", "data")
logPath = os.path.join("logs", d1)
logLevel = logging.INFO
consoleLog = 1


def display(fileName):
    loadModel = pickle.load(open(os.path.join(modelFolder, fileName), "rb"))
    if arguments.testfile:
        df = pd.read_csv(
            os.path.join(inputFolder, arguments.testfile + ".csv"),
        )
    else:
        df = pd.read_csv(os.path.join(inputFolder, "test.csv"))
    testX = df.drop("median_house_value", axis=1)
    testY = df["median_house_value"].copy()
    YPred = loadModel.predict(testX)
    mse = mean_squared_error(testY, YPred)
    rmse = np.sqrt(mse)
    logger.info("RMSE of " + fileName + " : " + str(rmse))


try:
    argumentParser = argparse.ArgumentParser(
        prog="test",
        description="testing the data",
    )
    argumentParser.add_argument(
        "--testfolder",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--testfile",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--modelfolder",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--log_level",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--log_path",
        action="store",
        type=str,
        required=False,
    )
    argumentParser.add_argument(
        "--no_console_log",
        action="store_true",
        required=False,
    )
    arguments = argumentParser.parse_args()
    if arguments.testfolder:
        inputFolder = os.path.join("datasets", arguments.testfolder)
    if arguments.modelfolder:
        modelFolder = os.path.join("artifacts", arguments.modelfolder)
    if arguments.log_path:
        logFolder = os.path.join(arguments.log_path, "ingestData.txt")
    print(arguments)
except Exception as e:
    print("Error occured")
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

    logFile = os.path.join(logPath, "test.log")

    if arguments.no_console_log:
        consoleLog = 0

    try:
        logging.basicConfig(filename=logFile, level=logLevel)
        logger = logging.getLogger("test")
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
    models = os.listdir(modelFolder)
    for m in models:
        display(m)
except Exception as e:
    logger.error("error occured")
    logger.error(e)
