import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineering:
    """ all pandas feature engineering methods """

    def __init__(self):
        pass

    def num_notifications_last_24h(self, merge_df: pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of notifications in the last 24 hours
        """
        def count_notifications_last_24h(group):
            return group["event_id"].rolling("24h").count() - 1  # subtract current row
        
        merge_df["num_notifications_last_24h"] = merge_df.groupby("user_id", group_keys=False).apply(count_notifications_last_24h)
        return merge_df
    
    def count_opens_last_7_days(self, group: pd.DataFrame) -> pd.DataFrame:
        # same as above for opens in 7 days
        group["opens_last_7_days"] = group["opened"].rolling("7d").sum()
        return group

    def calc_delay_since_last_open_notification(self, merge_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the delay since the last open notification
        """
        merge_df['total_hours'] = (merge_df['day'] * 24) + merge_df['hour']
        merge_df['last_open_time'] = merge_df['total_hours'].where(merge_df['opened'] == 1)
        merge_df['prev_open_timestamp'] = merge_df['last_open_time'].shift(1).ffill()
        merge_df['delay_since_last_open_notification'] = (
            merge_df['total_hours'] - merge_df['prev_open_timestamp']
        ).fillna(0)

        merge_df = merge_df.drop(columns=['total_hours', 'last_open_time', 'prev_open_timestamp'])

        return merge_df

    def calc_user_open_rate(self, merge_df: pd.DataFrame) -> pd.DataFrame:
        #user open rate = open notifi before t / send notifi before t
        x = merge_df.groupby("user_id")

        open = x["opened"].cumsum()
        send = x.cumcount()
        merge_df["user_open_rate"] = open.shift(1) / send.shift(1).replace(0, np.nan)

        return merge_df

    def calc_user_hour_open_rate(self, merge_df: pd.DataFrame) -> pd.DataFrame:
        #user open rate = open notifi at hour h before t / send notifi at hour h before t
        x = merge_df.groupby(["user_id", "hour"])

        open = x["opened"].cumsum()
        send = x.cumcount()
        merge_df["user_hour_open_rate"] = open.shift(1) / send.shift(1).replace(0, np.nan)

        return merge_df

    def calc_hour_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode hour as cyclical features using sin/cos transformation
        """
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline for feature engineering
    """
    feature_engineering = FeatureEngineering()

    df["timestamp"] = pd.to_datetime(datetime.now())  \
        + pd.to_timedelta(df["day"], unit="D") \
        + pd.to_timedelta(df["hour"], unit='h')
    df.set_index("timestamp", inplace=True)
    df.sort_values(by=["user_id", "timestamp"], inplace=True)

    df = feature_engineering.num_notifications_last_24h(df.copy())
    # df = feature_engineering.count_opens_last_7_days(df.copy())
    df = feature_engineering.calc_delay_since_last_open_notification(df.copy())
    df = feature_engineering.calc_user_open_rate(df.copy())
    df = feature_engineering.calc_user_hour_open_rate(df.copy())
    df = feature_engineering.calc_hour_cyclical(df.copy())

    return df

def apply_feature_engineering():
    """
    Apply feature engineering to the dataframe
    """
    path_to_data = "data/logs/training_data.csv"
    path_to_data_features = "data/training_data_features.csv"
    df = pd.read_csv(path_to_data)
    df = feature_engineering_pipeline(df)
    # df.to_csv(path_to_data_features, index=False)
    return df

if __name__ == "__main__":
    df = apply_feature_engineering()
    print(df.isna().sum())
    print(len(df))
