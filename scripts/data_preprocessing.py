import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, df):
        """
        Initialize the DataPreprocessor with the dataframe.
        :param df: Pandas DataFrame containing the dataset to preprocess
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
