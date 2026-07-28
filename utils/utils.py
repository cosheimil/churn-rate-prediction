import pandas as pd
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import joblib

# ========================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
def load_data(filename: str = "raw/telco_churn.csv") -> pd.DataFrame:
    file_path = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(file_path)
    return df

# ========================================================================
# Data Eyeballing: Display basic info about the DataFrame.
def data_checkup(df):
    num_data = []
    cat_data = []
    
    for col in df.columns:
        base_data = {
            'column': col,
            'num_rows': int(len(df)),
            'num_nulls': int(df[col].isnull().sum())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            num_data.append({
                **base_data,
                'min': round(min_val.item(), 4) if not pd.isna(min_val) else None,
                'max': round(max_val.item(), 4) if not pd.isna(max_val) else None,
                'mean': round(mean_val.item(), 4) if not pd.isna(mean_val) else None,
                'median': round(median_val.item(), 4) if not pd.isna(median_val) else None,
                'std': round(std_val.item(), 4) if not pd.isna(std_val) else None
            })
        else:
            cat_data.append({
                **base_data,
                'unique_options': df[col].unique().tolist()
            })
    
    df_numerical = pd.DataFrame(num_data)
    df_categorical = pd.DataFrame(cat_data)
    
    return df_numerical, df_categorical


# ========================================================================
# Multicollinearity check using Variance Inflation Factor (VIF)
def check_multicollinearity(df, threshold=5.0, include_intercept=False):
    num_df = df.select_dtypes(include=[float, int]).copy()
    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns found for VIF calculation")

    if include_intercept:
        num_df = num_df.assign(intercept=1.0)

    # Remove perfect collinear or NaN rows
    num_df = num_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    vif_list = []
    for i, col in enumerate(num_df.columns):
        vif_value = variance_inflation_factor(num_df.values, i)
        vif_list.append({'feature': col, 'VIF': round(vif_value, 4), 'high_vif': vif_value > threshold})

    vif_df = pd.DataFrame(vif_list).sort_values(by='VIF', ascending=False).reset_index(drop=True)
    return vif_df


# ========================================================================
# Check balance of target variable
def target_balance(df, target):
    vc = df[target].value_counts(dropna=False)
    return pd.DataFrame({
        "count": vc,
        "pct": vc / len(df)
    })


# ========================================================================
# Split data into train and test sets considering imbalanced data using stratified sampling
def split_data_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ========================================================================
# Data Processor
class PreprocessorManager:
    def __init__(self, processor_path=None):
        if processor_path is None:
            processor_path = os.path.join(os.path.dirname(__file__), "processor.pkl")
        self.processor_path = processor_path
        self.processor = None

    # Fit + Transform on TRAIN
    def fit_transform_train(self, X_train, numerical_cols=None, categorical_cols=None):
        X_train_processed = X_train.copy()
        self.processor = {}

        # ----- Numerical -----
        if numerical_cols:
            scaler = StandardScaler()
            X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            self.processor['scaler'] = scaler
            self.processor['numerical_columns'] = numerical_cols

        # ----- Categorical -----
        encoders = {}
        if categorical_cols:
            for col in categorical_cols:
                unique_values = X_train[col].unique()
                num_unique = len(unique_values)

                # Binary
                if set(unique_values) <= {0,1} or set(unique_values) <= {'0','1'}:
                    encoders[col] = None
                    continue

                # Label encoding for 2 unique values
                elif num_unique == 2:
                    le = LabelEncoder()
                    le.fit(sorted(unique_values))
                    X_train_processed[col] = le.transform(X_train[col])
                    encoders[col] = le

                # One-hot encoding for >2 unique values
                elif num_unique > 2:
                    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded = ohe.fit_transform(X_train[[col]])
                    new_cols = [f"{col}_{val}" for val in ohe.categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=X_train.index)
                    X_train_processed = pd.concat([X_train_processed.drop(columns=[col]), encoded_df], axis=1)
                    encoders[col] = ohe

            self.processor['encoders'] = encoders
            self.processor['categorical_columns'] = categorical_cols

        # Save processor
        joblib.dump(self.processor, self.processor_path)
        print(f"Processor saved to: {self.processor_path}")

        return X_train_processed




    # Transform only on TEST
    def transform_test(self, X_test):
        if self.processor is None:
            if not os.path.exists(self.processor_path):
                raise FileNotFoundError(f"Processor file not found at {self.processor_path}")
            self.processor = joblib.load(self.processor_path)

        X_test_processed = X_test.copy()

        # ----- Numerical -----
        if 'scaler' in self.processor:
            scaler = self.processor['scaler']
            num_cols = self.processor['numerical_columns']
            X_test_processed[num_cols] = scaler.transform(X_test[num_cols])

        # ----- Categorical -----
        if 'encoders' in self.processor:
            encoders = self.processor['encoders']
            for col, encoder in encoders.items():
                if encoder is None:
                    continue
                elif hasattr(encoder, "classes_"):  # LabelEncoder
                    X_test_processed[col] = encoder.transform(X_test[col])
                else:  # OneHotEncoder
                    encoded = encoder.transform(X_test[[col]])
                    new_cols = [f"{col}_{val}" for val in encoder.categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=X_test.index)
                    X_test_processed = pd.concat([X_test_processed.drop(columns=[col]), encoded_df], axis=1)

        return X_test_processed