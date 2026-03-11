import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class FeatureImputer:
    """
    Imputes missing valuees
    """

    def __init__(self):
        self.user_open_rate_imputer = SimpleImputer(strategy='mean')
        self.user_hour_open_rate_imputer = SimpleImputer(strategy='mean')
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fit imputers to get global averages
        """
        # Fit on non-null values to get global mean
        self.user_open_rate_imputer.fit(df[['user_open_rate']])
        self.user_hour_open_rate_imputer.fit(df[['user_hour_open_rate']])
        self.is_fitted = True
        
        print(f"Global user_open_rate mean: {self.user_open_rate_imputer.statistics_[0]:.4f}")
        print(f"Global user_hour_open_rate mean: {self.user_hour_open_rate_imputer.statistics_[0]:.4f}")
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by filling missing values with global averages
        """
        if not self.is_fitted:
            raise ValueError("not imputed, call fit() first.")
        
        df = df.copy()
        
        # Impute user_open_rate
        df['user_open_rate'] = self.user_open_rate_imputer.transform(
            df[['user_open_rate']]
        ).flatten()
        
        # Impute user_hour_open_rate
        df['user_hour_open_rate'] = self.user_hour_open_rate_imputer.transform(
            df[['user_hour_open_rate']]
        ).flatten()
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        """
        self.fit(df)
        return self.transform(df)


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to impute features in a dataframe
    """
    imputer = FeatureImputer()
    return imputer.fit_transform(df)


if __name__ == "__main__":
    from data_pipeline.feature_engg import apply_feature_engineering
    
    # Load data with engineered features
    df = apply_feature_engineering()
    
    print("Before imputation:")
    print(df[['user_open_rate', 'user_hour_open_rate']].isna().sum())
    
    # Impute missing values
    imputer = FeatureImputer()
    df_imputed = imputer.fit_transform(df)
    
    print("\nAfter imputation:")
    print(df_imputed[['user_open_rate', 'user_hour_open_rate']].isna().sum())
    
    # Save imputed data
    output_path = "data/training_data_features_imputed.csv"
    df_imputed.to_csv(output_path, index=True)
    print(f"\nSaved imputed data to {output_path}")
