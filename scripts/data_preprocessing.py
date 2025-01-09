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


def extract_datetime_features(self):
    """
    Extracts features from the 'Date' column.
    """
    self.df['Date'] = pd.to_datetime(self.df['Date'])

    # Extracting day of week, weekend flag, and quarter
    self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
    self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    self.df['Quarter'] = self.df['Date'].dt.quarter

    # Generate MonthPosition feature
    self.df['MonthPosition'] = self.df['Date'].dt.day.apply(
        lambda x: 'Start' if x <= 10 else ('Mid' if x <= 20 else 'End')
    )

    # One-hot encode MonthPosition
    self.df = pd.get_dummies(self.df, columns=['MonthPosition'], drop_first=True)


def encode_categorical_data(self):
    """
    Encodes categorical data:
    - Label encoding for specific ordinal columns.
    - One-hot encoding for nominal columns.
    """
    label_cols = ['StateHoliday', 'StoreType', 'Assortment']
    label_encoder = LabelEncoder()

    for col in label_cols:
        if col in self.df.columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])

    onehot_cols = ['DayOfWeek', 'Quarter']
    self.df = pd.get_dummies(self.df, columns=onehot_cols, drop_first=True)

def scale_numeric_features(self):
    """
    Scales numerical features using StandardScaler.
    """
    num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
    self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])

def feature_engineering(self):
    """
    Add features for holidays, promotions, and competitor data.
    """
    holidays = pd.to_datetime(['2024-12-25', '2024-01-01'])  # Example dates

    self.df['DaysToNextHoliday'] = self.df['Date'].apply(
        lambda x: min([(h - x).days for h in holidays if h >= x], default=np.nan)
    )
    self.df['DaysAfterLastHoliday'] = self.df['Date'].apply(
        lambda x: min([(x - h).days for h in holidays if h <= x], default=np.nan)
    )




