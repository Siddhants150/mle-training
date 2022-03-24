from setuptools import setup

setup(
    name="MLE Training Assignment House Price Prediction",
    version="0.2",
    scripts=["./src/ingestData.py", "./src/train.py", "./src/test.py"],
)
