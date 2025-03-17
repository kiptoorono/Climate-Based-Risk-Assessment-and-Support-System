import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def forecast_climate_variables(data_path, target_counties, target_variables, forecast_horizon=3, test_size=0.2, output_dir='./'):
    """
    Forecast climate variables using XGBoost
    
    Parameters:
    data_path (str): Path to CSV file containing extracted features
    target_counties (list): List of counties to forecast for
    target_variables (list): List of variables to forecast (rain, temp, soil)
    forecast_horizon (int): Number of time steps to forecast ahead
    test_size (float): Proportion of data to use for testing
    output_dir (str): Directory to save output plots
    
    Returns:
    dict: Dictionary of trained models and evaluation metrics
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Ensure Date column is datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
    
    results = {}
    
    for county in target_counties:
        county_results = {}
        
        for var_type in target_variables:
            print(f"\n{'='*60}")
            print(f"FORECASTING {county.upper()}_{var_type.upper()}")
            print(f"{'='*60}")
            
            target_col = f"{county}_{var_type}"
            if target_col not in data.columns:
                print(f"Warning: {target_col} not found in dataset. Skipping.")
                continue
            
            # Select relevant features
            # Features for this target and county
            useful_suffixes = ['_prev_month', '_prev_year', '_3month_avg', 
                              f'_zone_diff', '_anomaly', '_zscore']
            
            # Get zone for this county and related zone features
            try:
                zone_id = data[f'{county}_zone'].iloc[0]
                zone_cols = [col for col in data.columns if f'zone_{zone_id}_' in col and var_type in col]
            except:
                print(f"Warning: Zone information not found for {county}. Using alternative features.")
                zone_cols = []
            
            # Base features for this county
            county_feature_cols = [col for col in data.columns if 
                                col.startswith(county) and 
                                any(suffix in col for suffix in useful_suffixes)]
            
            # Temporal features
            time_cols = [col for col in data.columns if 
                        col in ['Month', 'Year', 'is_wet_season'] or 'Season' in col]
            
            # Combine all feature columns
            feature_cols = county_feature_cols + zone_cols + time_cols
            
            # Make sure we have some features
            if len(feature_cols) < 3:
                print(f"Not enough features found for {target_col}. Skipping.")
                continue
            
            # Create train/test split based on time
            split_idx = int(len(data) * (1 - test_size))
            X_train = data.iloc[:split_idx][feature_cols]
            y_train = data.iloc[:split_idx][target_col]
            X_test = data.iloc[split_idx:][feature_cols]
            y_test = data.iloc[split_idx:][target_col]
            
            print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Testing data: {X_test.shape[0]} samples")
            
            # Remove any columns with NaN values
            nan_cols = X_train.columns[X_train.isna().any()].tolist()
            if nan_cols:
                print(f"Removing {len(nan_cols)} columns with NaN values: {nan_cols}")
                X_train = X_train.drop(columns=nan_cols)
                X_test = X_test.drop(columns=nan_cols)
            
            # Ensure no remaining NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Initialize and train XGBoost model
            print("Training XGBoost model...")
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                random_state=42
            )
            
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
                early_stopping_rounds=20
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model Evaluation:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Feature importance
            if len(model.feature_importances_) > 0:
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10))
                
                # Plot feature importance
                plt.figure(figsize=(12, 6))
                top_features = feature_importance.head(15)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title(f'Feature Importance for {county.capitalize()} {var_type.capitalize()} Forecast')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{county}_{var_type}_importance.png")
                
            # Plot predictions vs actual
            plt.figure(figsize=(14, 6))
            plt.plot(data.iloc[split_idx:]['Date'], y_test.values, 'b-', label='Actual')
            plt.plot(data.iloc[split_idx:]['Date'], y_pred, 'r--', label='Predicted')
            plt.title(f'{county.capitalize()} {var_type.capitalize()} Forecast')
            plt.xlabel('Date')
            plt.ylabel(f'{var_type.capitalize()} Value')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{county}_{var_type}_forecast.png")
            
            # Store results
            county_results[var_type] = {
                'model': model,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actuals': y_test.values
            }
            
            print(f"Results saved to {output_dir}/{county}_{var_type}_forecast.png")
            
        results[county] = county_results
    
    return results

def generate_future_forecast(data, model, feature_cols, target_col, periods=3):
    """
    Generate future forecasts for the specified periods
    
    Parameters:
    data (pd.DataFrame): Input data with extracted features
    model (object): Trained forecasting model
    feature_cols (list): List of feature columns
    target_col (str): Target column to forecast
    periods (int): Number of future periods to forecast
    
    Returns:
    pd.DataFrame: DataFrame with forecasted values
    """
    # Sort data by date
    data = data.sort_values('Date').copy()
    
    # Create a new dataframe for future dates
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq='M')[1:]
    
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Add month, year features
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Year'] = future_df['Date'].dt.year
    
    # Add season information
    conditions = [
        (future_df['Month'].isin([3, 4, 5])),
        (future_df['Month'].isin([6, 7, 8, 9])),
        (future_df['Month'].isin([10, 11, 12])),
        (future_df['Month'].isin([1, 2]))
    ]
    season_values = ['long_rainy', 'long_dry', 'short_rainy', 'short_dry']
    future_df['Season'] = np.select(conditions, season_values)
    future_df['is_wet_season'] = future_df['Season'].isin(['long_rainy', 'short_rainy']).astype(int)
    
    # For each prediction period, we need to update lagged features
    forecasts = []
    current_data = data.copy()
    
    for i in range(periods):
        # Get the next date to predict
        next_date = future_dates[i]
        next_row = future_df[future_df['Date'] == next_date].copy()
        
        # Create lag features for this row
        county, var_type = target_col.split('_')
        prev_value = current_data.iloc[-1][target_col]
        
        # Add previous month value
        next_row[f'{target_col}_prev_month'] = prev_value
        
        # Add previous year value (12-month lag)
        if len(current_data) >= 12:
            next_row[f'{target_col}_prev_year'] = current_data.iloc[-12][target_col]
        else:
            # If we don't have enough history, use the mean
            next_row[f'{target_col}_prev_year'] = current_data[target_col].mean()
        
        # Add 3-month rolling average
        if len(current_data) >= 3:
            next_row[f'{target_col}_3month_avg'] = current_data.iloc[-3:][target_col].mean()
        else:
            next_row[f'{target_col}_3month_avg'] = current_data[target_col].mean()
        
        # For other complex features, we might need to simplify or approximate
        # This is a simplified approach - in practice, you'd need more sophisticated
        # handling of derived features
            
        # Select only the feature columns needed for prediction
        pred_features = [col for col in feature_cols if col in next_row.columns]
        missing_features = [col for col in feature_cols if col not in next_row.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing for future prediction.")
            print(f"Missing features: {missing_features[:5]}...")
        
        # Make prediction
        if len(pred_features) > 0:
            prediction = model.predict(next_row[pred_features])
            next_row[target_col] = prediction[0]
        else:
            # If we can't make a prediction, use the last known value
            next_row[target_col] = prev_value
            print("Warning: Not enough features for prediction. Using last known value.")
        
        # Add this prediction to the results
        forecasts.append(next_row)
        
        # Update current_data with this prediction for next iteration
        current_data = pd.concat([current_data, next_row[[col for col in current_data.columns if col in next_row.columns]]])
    
    # Combine all forecast periods
    future_forecast = pd.concat(forecasts)
    return future_forecast

if __name__ == "__main__":
    # Replace with your data path
    data_path = "E:\Agriculture project\Data\Processed\Merged_data_features.csv"
    output_dir = "E:\Agriculture project"
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define counties and variables to forecast
    target_counties = ['Embu', 'Kericho', 'Meru']
    target_variables = ['rain', 'temp']
    
    # Run forecasting
    results = forecast_climate_variables(
        data_path=data_path,
        target_counties=target_counties,
        target_variables=target_variables,
        output_dir=output_dir
    )
    
    print("\nForecasting complete!")