import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    def create_aggregate_features(self):
        """
        Creates aggregate features such as total, average, count, and standard deviation of transactions per customer.
        """
        self.df_agg = self.df.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('Amount', 'count'),
            Transaction_Amount_Std=('Amount', 'std')
        ).reset_index()
        return self.df_agg

    def extract_temporal_features(self):
        """
        Extracts temporal features such as hour, day, month, and year from the 'TransactionStartTime'.
        """
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year
        return self.df[['TransactionStartTime', 'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]

    def encode_categorical_variables(self, encoding_type='onehot'):
        """
        Encodes categorical variables using One-Hot Encoding or Label Encoding.
        """
        if encoding_type == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_columns = pd.DataFrame(ohe.fit_transform(self.df[['ProductCategory', 'ChannelId']]),
                                           columns=ohe.get_feature_names_out(['ProductCategory', 'ChannelId']))
            self.df = pd.concat([self.df, encoded_columns], axis=1)
            self.df.drop(['ProductCategory', 'ChannelId'], axis=1, inplace=True)
        elif encoding_type == 'label':
            le = LabelEncoder()
            self.df['ProductCategory_Encoded'] = le.fit_transform(self.df['ProductCategory'])
            self.df['ChannelId_Encoded'] = le.fit_transform(self.df['ChannelId'])
            self.df.drop(['ProductCategory', 'ChannelId'], axis=1, inplace=True)
        return self.df

    def handle_missing_values(self, strategy='mean'):
        """
        Handles missing values by imputing missing data in numerical columns only.
        Non-numeric columns are left unchanged or can be imputed separately if needed.
        """
        # Select numerical columns for imputation
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        # Apply SimpleImputer only to numerical columns
        imputer = SimpleImputer(strategy=strategy)
        
        # Impute missing values for numeric columns
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        
        return self.df
    

    
    def normalize_or_standardize(self, method='normalize'):
        """
        Normalizes or standardizes numerical features.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        if method == 'normalize':
            scaler = MinMaxScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        elif method == 'standardize':
            scaler = StandardScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        return self.df
