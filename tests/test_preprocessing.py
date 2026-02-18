import os
import pandas as pd


def test_preprocessed_files_exist():
    assert os.path.exists("X_train.parquet"), "X_train.parquet missing"
    assert os.path.exists("X_test.parquet"), "X_test.parquet missing"
    assert os.path.exists("y_train.parquet"), "y_train.parquet missing"
    assert os.path.exists("y_test.parquet"), "y_test.parquet missing"


def test_preprocessed_shapes():
    X_train = pd.read_parquet("X_train.parquet")
    X_test = pd.read_parquet("X_test.parquet")
    y_train = pd.read_parquet("y_train.parquet")
    y_test = pd.read_parquet("y_test.parquet")

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
