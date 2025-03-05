import pandas as pd
import numpy as np

# Load data (assuming Excel format with 'Date' column)
df = pd.read_excel('E:/Agriculture project/Data/Semi-prepared data/Rainfall_total.xlsx', parse_dates=['Date'], index_col='Date')

# Statistical Features
features = pd.DataFrame()
features['mean_rainfall'] = df.mean()
features['median_rainfall'] = df.median()
features['std_rainfall'] = df.std()
features['cv_rainfall'] = df.std() / df.mean()  # Coefficient of Variation
features['skewness'] = df.skew()
features['kurtosis'] = df.kurtosis()
features['min_rainfall'] = df.min()
features['max_rainfall'] = df.max()

# Temporal Features
monthly_avg = df.groupby(df.index.month).mean()
yearly_totals = df.resample('Y').sum()
rolling_avg = df.rolling(window=3).mean()

# Extreme Weather Features
low_threshold = df.quantile(0.1)
high_threshold = df.quantile(0.9)
features['drought_months'] = (df < low_threshold).sum()
features['heavy_rain_months'] = (df > high_threshold).sum()
features['zero_rainfall_months'] = (df == 0).sum()

# Rainfall Anomalies
long_term_mean = df.mean()
anomalies = df - long_term_mean

# Save extracted features
features.to_excel('rainfall_features.xlsx')
print("Feature extraction complete. Data saved to rainfall_features.xlsx")
