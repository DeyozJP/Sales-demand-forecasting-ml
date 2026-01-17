import pandas as pd
from datetime import datetime


# create a function to determine if the dates in each time series are continous
def is_date_continous(
        dataframe: pd.DataFrame,
        date_column: str,
        group_columns: list = None
    ) -> pd.DataFrame:
    """
    Check if the dates in the series are contious.
    Arguments:
    - dataframe: pd.DataFrame, input dataframe
    - date_column: str, name of the date column
    - group_columns (optional): list, list of the columns to group by
    """
        
    missing_dates_list = [] # dictionary to store the number of missing dates per time series

    for k, g in dataframe.groupby(group_columns):
        date_range = pd.date_range(
            start=g[date_column].min(),
            end=g[date_column].max(),
            freq="D"
            )

        missing_dates = date_range.difference(pd.to_datetime(g[date_column]))

        for d in missing_dates:
            missing_dates_list.append({
                **dict(zip(group_columns, k if isinstance(k, tuple) else(k, ))),
                "missing_date": d
            })

        
    missing_dates_df = pd.DataFrame(missing_dates_list)
    if missing_dates_df.empty:
        print("No missing dates found in the time series.")
        return missing_dates_df
        
    else:
        print(f"Found {missing_dates_df.shape[0]} missing dates in the time series.")
        return missing_dates_df



    
    
def calculate_rolling_dates(test_cuttoff_date: str | datetime, horizon:int=7, n_folds: int=5):
    """
    Generates the rollig dates for walf forward validation.

    Arguments:
    - test_cuttoff_date: str: The date used to determine the cutoff for the test set.
    - horizon: The number of days in the future to predict.
    - n_folds: The number of folds for cross-validation.
    Returns:
    - rolling_dates: pd.DatetimeIndex: The rolling dates for walf forward validation.
    """
    # Calculate the last date for the training set 
    try:
        if isinstance(test_cuttoff_date, str):
            test_cuttoff_date = pd.to_datetime(test_cuttoff_date)
    except Exception as e:
        raise ValueError("test_cuttoff_date must be a string or pd.datetime") from e

    last_train_date = test_cuttoff_date - pd.Timedelta(days=horizon)
    start_date = test_cuttoff_date - pd.Timedelta(days = horizon * n_folds)

    rolling_dates = pd.date_range(
        start=start_date,
        end=last_train_date,
        freq=f'{horizon}D'
    )
    return rolling_dates




# Write a function to get a features for the traininig set.

# Function to create a lag features
def create_lag_features(dataframe: pd.DataFrame, 
                       date_col:str, 
                       lag_days: list | int, 
                       target_col: str, 
                       group_column_id: list=None,) -> pd.DataFrame:
    
    try:
        df = dataframe.copy() # Copy the dataframe to avoid the modifying the original dataframe
       
        if isinstance(lag_days, int):
            lag_days = [lag_days]

        for lag in lag_days:
            if group_column_id is not None:
                df[f"{target_col}_lag_{lag}"] = df.groupby(group_column_id)[target_col].shift(lag)
                df = df.sort_values(by=group_column_id + [date_col])
            else:
                df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)


        return df
    except Exception as e:
        print(f"An error occurred while creating lag features: {e}")
        raise e

import pandas as pd

def create_custom_features(
    df: pd.DataFrame,
    group_columns_id: list,
    target_column: str,
    date_column: str
) -> pd.DataFrame:
    """
    Create time-series features for each group (SKU) in the dataframe.

    Features:
    - price_index: current price / historical expanding mean price
    - relative_yesterday_sales_to_mean: yesterday's sales relative to historical expanding mean
    - weekly_ma_units_sold: 7-day moving average of units sold
    - monthly_ma_units_sold: 30-day moving average of units sold
    - sales_relative_to_weekly_avg / sales_relative_to_monthly_avg
    - sell_through_rate_yesterday: yesterday's units_sold / yesterday's stock_available

    Parameters:
    - df: input dataframe
    - group_columns_id: list of columns to group by (e.g., ['sku'])
    - target_column: target variable to forecast (e.g., 'units_sold')
    - date_column: date column for sorting (e.g., 'date')

    Returns:
    - df: dataframe with new features
    """

    df = df.copy()

    # ------------------------------
    # Price index: current price / historical expanding mean price per SKU
    # ------------------------------
    # df['price_index'] = df.groupby(group_columns_id)['price_unit']\
                        #   .transform(lambda x: x.shift(1).expanding().mean())
    # df['price_index'] = df['price_unit'] / df['price_index']

    # df.drop(columns=['price_unit'], inplace=True) # We created price_index, so we can drop price_unit

    # ------------------------------
    # Relative yesterday's sales to historical expanding mean per SKU
    # ------------------------------
    # df['relative_yesterday_sales_to_mean'] = df.groupby(group_columns_id)[target_column]\
                                            #    .transform(lambda x: x.shift(1) / x.shift(1).expanding().mean())

    # ------------------------------
    # Weekly and monthly moving averages of units_sold per SKU
    # ------------------------------
    df['weekly_ma_units_sold'] = df.groupby(group_columns_id)[target_column]\
                                   .transform(lambda x: x.shift(1).rolling(window=7).mean())
    # 
    df['monthly_ma_units_sold'] = df.groupby(group_columns_id)[target_column]\
                                    .transform(lambda x: x.shift(1).rolling(window=30).mean())

    # ------------------------------
    # Sales relative to rolling averages
    # ------------------------------
    # df['sales_relative_to_weekly_avg'] = df.groupby(group_columns_id)[target_column].shift(1) / df['weekly_ma_units_sold']
    # df['sales_relative_to_monthly_avg'] = df.groupby(group_columns_id)[target_column].shift(1) / df['monthly_ma_units_sold']

    # ------------------------------
    # Sell-through rate yesterday
    # ------------------------------
    df['sell_through_rate_yesterday'] = df.groupby(group_columns_id)[target_column].shift(1) / \
                                       df.groupby(group_columns_id)['stock_available'].shift(1)

    # ------------------------------
    # Sort dataframe by group + date
    # ------------------------------
    df = df.sort_values(by=group_columns_id + [date_column]).reset_index(drop=True)
    df.dropna(inplace=True)

    return df


# Define a function to create a temporal features 
def create_temporal_features(
        dataframe: pd.DataFrame, 
        date_col: str,
    
    ) -> pd.DataFrame:

    # Copy the dataframe to avoid modifying the original dataframe
    df = dataframe.copy()

    # Extract temporal features from the date_col
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month


    return df

def make_features_from_history(
        history_df: pd.DataFrame,
        date_col=str, 
        features_to_use_in_validation=list
        ) -> pd.DataFrame:
    history_df = history_df.sort_values(by=date_col)

    features_dict = {}

    for feature in features_to_use_in_validation:
        features_dict[feature] = [history_df[feature].iloc[-1]]
    

    return pd.DataFrame(features_dict)



    




def test_train_validation_split(training_set, validation_set, origin_date):
    assert training_set['date'].max() < origin_date, ("Training set contains dates after the origin date.")

    assert validation_set['date'].min() >= origin_date, ("Validation set contains dates before the origin date.")
