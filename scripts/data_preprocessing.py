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

def handle_missing_values(self):
    """
    Handles missing values:
    - Drops columns with more than 50% missing data.
    - Imputes remaining missing values with median for numerical and mode for categorical data.
    """
    missing_threshold = 0.5
    missing_percent = self.df.isnull().mean()
    self.df.drop(columns=missing_percent[missing_percent > missing_threshold].index, inplace=True)

    # Impute remaining missing values
    num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = self.df.select_dtypes(include=['object']).columns

    self.df[num_cols] = SimpleImputer(strategy='median').fit_transform(self.df[num_cols])
    self.df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(self.df[cat_cols])
