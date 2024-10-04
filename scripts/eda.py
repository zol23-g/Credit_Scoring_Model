import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.logger import setup_logger  # Assuming you have a logger setup script


class EDA:
    def __init__(self, df):
        self.df = df
        self.logger = setup_logger()  # Setup the logger to track progress

    def overview(self):
        """
        Provides an overview of the dataset: shape, columns, and data types.
        """
        self.logger.info("Generating dataset overview...")
        overview = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "data_types": self.df.dtypes
        }
        return overview

    def summary_statistics(self):
        """
        Calculates and returns summary statistics for the dataset.
        """
        self.logger.info("Calculating summary statistics...")
        return self.df.describe()

    def data_types(self):
        """
        Checks and returns data types of the columns.
        """
        self.logger.info("Checking data types...")
        return self.df.dtypes

    def plot_distribution(self, column):
        """
        Plots the distribution of a specified numerical column.
        """
        self.logger.info(f"Plotting distribution for {column}...")
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix for numerical columns.
        """
        self.logger.info("Calculating and plotting the correlation matrix...")
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64'])
        correlation = numeric_cols.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_categorical_distribution(self, column):
        """
        Plots the distribution of a specified categorical column.
        """
        self.logger.info(f"Plotting categorical distribution for {column}...")
        plt.figure(figsize=(10, 6))
        sns.countplot(y=self.df[column])
        plt.title(f'Distribution of {column}')
        plt.show()

    def check_missing_values(self):
        """
        Checks and returns missing values for each column in the dataset.
        """
        self.logger.info("Checking missing values...")
        missing_data = self.df.isnull().sum()
        return missing_data[missing_data > 0]

    def detect_outliers(self, column):
        """
        Detects outliers in the specified numerical column using a box plot.
        """
        self.logger.info(f"Detecting outliers for {column}...")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column])
        plt.title(f'Box Plot of {column}')
        plt.show()

    def visualize_distributions(self):
        """
        Visualizes distributions for all numerical and categorical columns in the dataset.
        """
        self.logger.info("Visualizing distributions for numerical and categorical columns...")

        # Plot distributions for numerical columns
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            self.plot_distribution(column)

        # Plot distributions for categorical columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.plot_categorical_distribution(column)

    def box_plots(self):
        """
        Creates box plots for all numerical columns for outlier detection.
        """
        self.logger.info("Creating box plots for outlier detection...")
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[col])
            plt.title(f'Box plot of {col}')
            plt.show()
