import pandas as pd
import os
import argparse
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np

modelFolder = os.path.join('artifacts', 'models')
inputFolder = os.path.join('datasets', 'data')

def display(fileName):
    loadModel = pickle.load(open(os.path.join(modelFolder, fileName), 'rb'))
    if arguments.testfile:
        df = pd.read_csv(os.path.join(inputFolder, arguments.testfile + '.csv'))
    else:
        df = pd.read_csv(os.path.join(inputFolder, 'test.csv'))
    testX = df.drop('median_house_value', axis = 1)
    testY = df['median_house_value'].copy()
    YPred = loadModel.predict(testX)
    mse = mean_squared_error(testY, YPred)
    rmse = np.sqrt(mse)
    print("RMSE of " + fileName + " : " + str(rmse))

try:
    argumentParser = argparse.ArgumentParser(prog = 'test', description = 'testing the data')
    argumentParser.add_argument('--testfolder', action = 'store', type = str, required = False)
    argumentParser.add_argument('--testfile', action = 'store', type = str, required = False)
    argumentParser.add_argument('--modelfolder', action = 'store', type = str, required = False)
    arguments = argumentParser.parse_args()
    if arguments.testfolder:
        inputFolder = os.path.join('datasets', arguments.testfolder)
    if arguments.modelfolder:
        modelFolder = os.path.join('artifacts', arguments.modelfolder)
    print(arguments)
except Exception as e:
    print("Error occured")
    print(e)
try:
    models = os.listdir(modelFolder)
    for m in models:
        display(m)
except Exception as e:
    print("error occured")
    print(e)