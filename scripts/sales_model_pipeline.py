import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib  # For saving and loading models
from datetime import datetime

class SalesModel:
    """
    A class to preprocess data and train a RandomForestRegressor model using sklearn pipelines.
    
    Attributes
    ----------
    model_pipeline : sklearn Pipeline
        The pipeline that preprocesses the data and fits the model.
    X_train : pd.DataFrame
        Training features after preprocessing.
    X_test : pd.DataFrame
        Testing features after preprocessing.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Testing target values.
        
    Methods
    -------
    preprocess_data(data, target_column, test_size=0.2, random_state=42):
        Preprocesses the data by scaling features in train and test sets separately.
    train_model():
        Trains the RandomForestRegressor model using the preprocessed training data.
    evaluate_model():
        Evaluates the trained model on the test data and returns the RMSE.
    tune_model(param_grid):
        Performs hyperparameter tuning using GridSearchCV.
    save_model():
        Saves the trained model to a file with a timestamp.
    load_model(filename):
        Loads a trained model from a file.
    feature_importance():
        Returns the feature importance from the trained model.
    plot_actual_vs_predicted():
        Plots the actual vs predicted values for the test set with enhanced visuals.
    make_predictions(test_data):
        Makes predictions on the provided test data.
    create_submission_file(test_data, submission_file_path):
        Creates a submission file for Kaggle using predictions from the test data.
    """

    def __init__(self):
        """Initializes the SalesModel class with a RandomForestRegressor pipeline."""
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize the features
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=64,
                min_samples_split=10,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ))
        ])
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self, data, target_column, test_size=0.2, random_state=42):
        """
        Preprocesses the data by splitting it into training and testing sets and scaling features.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing both features and the target column.
        target_column : str
            The name of the target column.
        test_size : float
            Proportion of the dataset to include in the test split (default is 0.2).
        random_state : int
            Random seed for reproducibility (default is 42).
        """
        # Split data into features (X) and target (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    def train_model(self):
        """Trains the RandomForestRegressor model using the preprocessed training data."""
        self.model_pipeline.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the trained model on the test data and returns the Root Mean Squared Error (RMSE).
        
        Returns
        -------
        float
            The Root Mean Squared Error of the model on the test data.
        """
        y_pred = self.model_pipeline.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        rmsle = self.rmsle(self.y_test, y_pred)
        
        print(f"Model RMSE: {rmse:.2f}")
        print(f"Model RMSLE: {rmsle:.4f}")

    def rmsle(self, y_true, y_pred):
        """Calculates Root Mean Squared Logarithmic Error (RMSLE)."""
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

    def tune_model(self, param_grid):
        """
        Performs hyperparameter tuning using GridSearchCV.
        
        Parameters
        ----------
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter settings to try.
        
        Returns
        -------
        dict
            The best parameters found during tuning.
        """
        grid_search = GridSearchCV(self.model_pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.model_pipeline = grid_search.best_estimator_
        return grid_search.best_params_

    def save_model(self):
        """Saves the trained model to a file with the current timestamp in the filename."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"sales_model_{timestamp}.pkl"
        joblib.dump(self.model_pipeline, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename):
        """Loads a trained model from a file."""
        self.model_pipeline = joblib.load(filename)

    def feature_importance(self):
        """Returns the feature importance from the trained model."""
        importances = self.model_pipeline.named_steps['model'].feature_importances_
        feature_names = self.X_train.columns
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def plot_actual_vs_predicted(self):
        """Plots the actual vs predicted values for the test set with enhanced visuals."""
        y_pred = self.model_pipeline.predict(self.X_test)
        
        plt.figure(figsize=(12, 6))
        sns.set_palette("Set2")
        sns.scatterplot(x=self.y_test, y=y_pred, alpha=0.6, s=100, edgecolor='w', linewidth=0.5)

        plt.plot([min(self.y_test), max(self.y_test)], 
                 [min(self.y_test), max(self.y_test)], 
                 color='darkorange', linestyle='--', linewidth=2, label='Ideal Prediction')

        plt.title('Actual vs Predicted Sales', fontsize=18)
        plt.xlabel('Actual Sales', fontsize=14)
        plt.ylabel('Predicted Sales', fontsize=14)
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.gca().set_facecolor('lightgrey')
        plt.tight_layout()
        plt.show()

    def make_predictions(self, test_data):
        """Makes predictions on the provided test data."""
        return self.model_pipeline.predict(test_data)

    def create_submission_file(self, test_data, submission_file_path):
        """
        Creates a submission file for Kaggle using predictions from the test data.
        
        Parameters
        ----------
        test_data : pd.DataFrame
            The dataset used to make predictions, should contain necessary identifiers.
        submission_file_path : str
            The file path where the submission CSV will be saved.
        """
        predictions = self.make_predictions(test_data)
        submission_df = pd.DataFrame({
            'Id': test_data.reset_index().Id,  # Assuming 'Id' exists in test_data
            'Sales': predictions
        })
        submission_df.to_csv(submission_file_path, index=False)
        print(f"Submission file saved as {submission_file_path}")