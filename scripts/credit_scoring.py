# scripts/credit_scoring.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CreditScoringModel:
    def __init__(self, df):
        self.df = df

    def calculate_rfms(self):
        """
        Calculates Recency, Frequency, Monetary, and Size (RFMS) scores for each customer.
        """
        self.df['TransactionDate'] = pd.to_datetime(self.df['TransactionStartTime'])
        recency_df = self.df.groupby('CustomerId').agg(Recency=('TransactionDate', lambda x: (x.max() - x.min()).days))
        frequency_df = self.df.groupby('CustomerId').agg(Frequency=('TransactionId', 'count'))
        monetary_df = self.df.groupby('CustomerId').agg(Monetary=('Amount', 'sum'))
        size_df = self.df.groupby('CustomerId').agg(Size=('Amount', 'mean'))
        rfms_df = pd.concat([recency_df, frequency_df, monetary_df, size_df], axis=1)
        return rfms_df

    def visualize_rfms_space(self, rfms_df):
        sns.pairplot(rfms_df)
        plt.suptitle('RFMS Space Visualization', y=1.02)
        plt.show()

    def classify_users(self, rfms_df):
        rfms_df['RiskScore'] = (
            (rfms_df['Recency'] < rfms_df['Recency'].mean()) &
            (rfms_df['Frequency'] > rfms_df['Frequency'].mean()) &
            (rfms_df['Monetary'] > rfms_df['Monetary'].mean()) &
            (rfms_df['Size'] > rfms_df['Size'].mean())
        ).astype(int)
        
        rfms_df['RiskLabel'] = rfms_df['RiskScore'].apply(lambda x: 'good' if x == 1 else 'bad')
        return rfms_df

    def calculate_woe_iv(self, df, feature, target):
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV) for a single feature.
        """
        # Create bins for the feature
        df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
        
        # Calculate the number of good (1) and bad (0) labels per bin
        grouped = df.groupby('bin')[target].agg(['count', 'sum'])
        grouped['bad'] = grouped['count'] - grouped['sum']  # bad is the complement of good
        
        # Calculate the distribution of goods and bads per bin
        grouped['good_dist'] = grouped['sum'] / grouped['sum'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        
        # Calculate WoE
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist']).replace([np.inf, -np.inf], 0)  # Replace infinities
        
        # Calculate IV
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        iv = grouped['iv'].sum()
        
        return grouped[['woe']], iv

    def calculate_information_value(self, rfms_df, target_column='RiskLabel'):
        """
        Calculate IV for all RFMS features.
        """
        rfms_df['RiskLabel'] = rfms_df['RiskLabel'].map({'good': 1, 'bad': 0})
        iv_values = {}
        for feature in ['Recency', 'Frequency', 'Monetary', 'Size']:
            _, iv = self.calculate_woe_iv(rfms_df, feature, 'RiskLabel')
            iv_values[feature] = iv
        
        print("Information Value (IV) for each feature:")
        for feature, iv in iv_values.items():
            print(f"{feature}: {iv}")
        return iv_values
