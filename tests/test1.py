import numpy as np
import pandas as pd
import pytest

def test_shape():
    xTrain = pd.read_csv('./datasets/data/train.csv')
    xTest = pd.read_csv('./datasets/data/test.csv')
    print("Running Test")
    assert xTrain.shape[1] == xTest.shape[1]
    print("Test successful")
    