import os
import pandas as pd
import numpy as np
import sklearn
import tarfile
import argparse
import logging
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from six.moves import urllib
from sklearn.impute import SimpleImputer
from datetime import date

today = date.today()

d1 = today.strftime("%d/%m/%Y")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
saveTrainTestPath = os.path.join('datasets', 'data')
logPath = os.path.join('logs', d1)
logLevel = logging.INFO
consoleLog = 1


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    logger.debug("Inside the fetch housing data function")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    logger.debug("Inside the load house data function")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def income_cat_proportions(data):
    logger.debug("Inside the income cat function")
    return data["income_cat"].value_counts() / len(data)

def data_prepared(data, fileName = 'train'):
    logger.debug("Inside the data prepared function")
    dataX = data.drop('median_house_value', axis = 1)
    dataY = data['median_house_value'].copy()
    dataNum = dataX.drop('ocean_proximity', axis = 1)
    oceanProximityColumn = dataX['ocean_proximity'].copy()
    simpleImputer = SimpleImputer(strategy = "median")
    simpleImputer.fit(dataNum)
    dataTr = simpleImputer.transform(dataNum)
    dataTr = pd.DataFrame(dataTr, columns=dataNum.columns, index = dataNum.index)
    dataTr["rooms_per_household"] = dataTr["total_rooms"] / dataTr["households"]
    dataTr["bedrooms_per_room"] = dataTr["total_bedrooms"] / dataTr["total_rooms"]
    dataTr["population_per_household"] = dataTr["population"] / dataTr["households"]
    dataPrepared = dataTr.join(pd.get_dummies(oceanProximityColumn, drop_first=True))
    dataPrepared['median_house_value'] = dataY
    logger.debug("data from " + fileName)
    logger.info(dataPrepared.head())
    try:
        if not os.path.isdir(saveTrainTestPath):
            os.makedirs(saveTrainTestPath)
        dataPrepared.to_csv(os.path.join(saveTrainTestPath, fileName + '.csv'))
    except Exception as e:
        logger.error('Error occured')
        logger.error(e)
    
try:
    argumentParser = argparse.ArgumentParser(prog = 'ingestData', description = 'Ingesting Data')
    argumentParser.add_argument('--folder', action = 'store', type = str, required = False)
    argumentParser.add_argument('--trainfile', action = 'store', type = str, required = False)
    argumentParser.add_argument('--testfile', action = 'store', type = str, required = False)
    argumentParser.add_argument('--log_level', action = 'store', type = str, required = False)
    argumentParser.add_argument('--log_path', action = 'store', type = str, required = False)
    argumentParser.add_argument('--no_console_log', action = 'store_true',  required = False)
    arguments = argumentParser.parse_args()
    if arguments.folder:
        saveTrainTestPath = os.path.join('datasets', arguments.folder)
    if arguments.log_path:
        logFolder = os.path.join(arguments.log_path, 'ingestData.txt')

    print(arguments)

except Exception as e:
    print("Error Occured")
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
    
    logFile = os.path.join(logPath, "ingestData.log")

    if arguments.no_console_log:
        consoleLog = 0

    try:
        logging.basicConfig(filename = logFile, level = logLevel)
        logger = logging.getLogger("ingestData")
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
    fetch_housing_data()
    housing = load_housing_data()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    logger.debug("Strat Train and Test Set Generated")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    logger.debug('Train and Test Set Generated')

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    logger.info('compare props')
    logger.info(compare_props)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    if arguments.trainfile:
        data_prepared(strat_train_set, arguments.trainfile)
    else:
        data_prepared(strat_train_set)
    
    if arguments.testfile:
        data_prepared(strat_test_set, arguments.testfile)
    else:
        data_prepared(strat_test_set, 'test')
    # data_prepared(strat_train_set)
    # data_prepared(strat_test_set, "test")

except Exception as e:
    logger.error("Error Occured")
    logger.error(e)
