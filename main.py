# main.py
"""Time Series Regression (Trend + Seasonality)
- Loads data/data/train.csv (user should download and place here)
- Extracts time features
- Trains baseline Linear Regression and Polynomial trend + seasonal model
- Prints R2 and RMSE and sample predictions
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "train.csv")

def load_data(path=DATA_PATH, nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download 'train.csv' from the Kaggle competition and place it in the data/ folder.")
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else None, nrows=nrows)
    return df

def create_time_features(df, date_col='date'):
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['weekday'] = df[date_col].dt.weekday
    # cyclical encoding for month and weekday
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    return df

def add_holiday_indicator(df, date_col='date', holidays=None):
    df = df.copy()
    if holidays is None:
        # sample list: user may replace with official holiday dates for their country
        holidays = []
    df['is_holiday'] = df[date_col].dt.normalize().isin(pd.to_datetime(holidays))
    df['is_holiday'] = df['is_holiday'].astype(int)
    return df

def prepare_features(df, target_col='sales'):
    # This prepare step assumes a 'date' column and a sales-like target column.
    df = create_time_features(df, 'date')
    df = add_holiday_indicator(df, 'date')
    feature_cols = ['day','month','year','weekday','month_sin','month_cos','weekday_sin','weekday_cos','is_holiday']
    # If extra numeric features exist (open, promo, etc.), include common ones
    for extra in ['store_nbr','item_nbr','onpromotion','open','promo']:
        if extra in df.columns:
            feature_cols.append(extra)
    X = df[feature_cols].fillna(0)
    y = df[target_col] if target_col in df.columns else None
    return X, y

def evaluate_model(model, X_test, y_test, name='model'):
    preds = model.predict(X_test)
    # compute RMSE in a version-compatible way
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return preds, rmse, r2


def baseline_linear(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def polynomial_trend_seasonal(X_train, y_train, degree=2):
    # We'll apply polynomial features to time components (day, month, year)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    time_cols = ['day','month','year']
    poly_part = poly.fit_transform(X_train[time_cols])
    # combine with the rest of features (seasonality cyclical + holiday + extras)
    other_cols = [c for c in X_train.columns if c not in time_cols]
    X_comb = np.hstack([poly_part, X_train[other_cols].values])
    model = LinearRegression()
    model.fit(X_comb, y_train)
    return model, poly, other_cols

def main():
    print("Time Series Regression (Trend + Seasonality) - main.py\n")
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Small sample to keep runtime reasonable for demonstration; remove nrows in real runs
    # Prepare data
    X, y = prepare_features(df, target_col='sales' if 'sales' in df.columns else ( 'units' if 'units' in df.columns else df.columns[-1] ))
    if y is None:
        print("Target column not found. Please ensure your dataset has a target column named 'sales' or 'units'.")
        return

    # Train/test split by time: keep last 20% as test set
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}\n")

    # Baseline linear regression
    lr = baseline_linear(X_train, y_train)
    evaluate_model(lr, X_test, y_test, name='Baseline Linear Regression')

    # Polynomial trend + seasonal model
    poly_degree = 2
    poly_model, poly_obj, other_cols = polynomial_trend_seasonal(X_train, y_train, degree=poly_degree)
    # Transform test set
    poly_part_test = poly_obj.transform(X_test[['day','month','year']])
    X_test_comb = np.hstack([poly_part_test, X_test[other_cols].values])
    preds, rmse, r2 = evaluate_model(poly_model, X_test_comb, y_test, name=f'Poly(deg={poly_degree}) + Seasonality')

    # Show a few sample predictions vs actuals
    print('\nSample predictions vs actuals:')
    sample_df = pd.DataFrame({
        'date': df['date'].iloc[split_idx:split_idx+10].astype(str).values,
        'actual': y_test.iloc[:10].values,
        'predicted': np.round(preds[:10], 3)
    })
    print(sample_df.to_string(index=False))

if __name__ == '__main__':
    main()
