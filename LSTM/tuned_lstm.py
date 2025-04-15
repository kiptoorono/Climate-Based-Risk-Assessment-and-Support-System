import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import math

# Create output directories
os.makedirs("tuned_forecasts", exist_ok=True)
os.makedirs("tuned_plots", exist_ok=True)
os.makedirs("tuned_evaluations", exist_ok=True)

# Load and prepare data
df = pd.read_csv("E:\Agriculture project\Data\Processed\Merged_data_features.csv", parse_dates=["Date"])
df.columns = df.columns.str.strip()
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index, errors="coerce")
df = df[df.index.notna()]

# Parameters
forecast_horizon = 60
sequence_length = 36  # Increased from 24 to capture longer patterns

# Identify unique regions
regions = {col.split('_')[0] for col in df.columns if col.endswith(("_rain", "_soil", "_temp"))}

for region in regions:
    rain_col, soil_col, temp_col = f"{region}_rain", f"{region}_soil", f"{region}_temp"
    if rain_col not in df or soil_col not in df or temp_col not in df:
        print(f"Skipping {region}: Missing data")
        continue

    # Select relevant features for the region
    region_features = [
        rain_col, soil_col, temp_col,  # Basic features
        f"{region}_rain_prev_month",    # Previous month's rainfall
        f"{region}_rain_prev_year",     # Previous year's rainfall
        f"{region}_rain_3month_avg",    # 3-month average
        f"{region}_rain_anomaly",       # Rainfall anomaly
        f"{region}_rain_zscore",        # Z-score for rainfall
        f"{region}_drought_index",      # Drought conditions
        f"{region}_drought_severity",   # Drought severity
        f"{region}_rain_consec_anomaly" # Consecutive anomalies
    ]
    
    # Filter out any missing features
    available_features = [f for f in region_features if f in df.columns]
    region_data = df[available_features].dropna()
    
    if len(region_data) < sequence_length + forecast_horizon:
        print(f"Skipping {region}: Not enough data")
        continue

    # Scale data with separate scalers for each feature type
    scalers = {}
    scaled_data = np.zeros_like(region_data)
    
    # Scale each feature type separately
    for i, col in enumerate(region_data.columns):
        scaler = MinMaxScaler()
        scaled_data[:, i] = scaler.fit_transform(region_data[[col]]).ravel()
        scalers[col] = scaler

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, :3])  # Only predict rain, soil, temp

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model with attention to rainfall features
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3)  # Output layer for rain, soil, temp
    ])

    # Custom loss function with dynamic weighting
    def custom_loss(y_true, y_pred):
        # Calculate individual losses
        rain_mse = tf.keras.backend.mean(tf.keras.backend.square(y_true[:, 0] - y_pred[:, 0]))
        soil_mse = tf.keras.backend.mean(tf.keras.backend.square(y_true[:, 1] - y_pred[:, 1]))
        temp_mse = tf.keras.backend.mean(tf.keras.backend.square(y_true[:, 2] - y_pred[:, 2]))
        
        # Weight rainfall errors more heavily
        return 1.5 * rain_mse + soil_mse + temp_mse

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=custom_loss)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model on test data
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions for each variable
    y_pred_rain = scalers[rain_col].inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_pred_soil = scalers[soil_col].inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_pred_temp = scalers[temp_col].inverse_transform(y_pred[:, 2].reshape(-1, 1))
    
    # Inverse transform actual values
    y_test_rain = scalers[rain_col].inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_test_soil = scalers[soil_col].inverse_transform(y_test[:, 1].reshape(-1, 1))
    y_test_temp = scalers[temp_col].inverse_transform(y_test[:, 2].reshape(-1, 1))
    
    # Calculate metrics
    rain_rmse = math.sqrt(mean_squared_error(y_test_rain, y_pred_rain))
    rain_mae = mean_absolute_error(y_test_rain, y_pred_rain)
    rain_r2 = r2_score(y_test_rain, y_pred_rain)
    
    soil_rmse = math.sqrt(mean_squared_error(y_test_soil, y_pred_soil))
    soil_mae = mean_absolute_error(y_test_soil, y_pred_soil)
    soil_r2 = r2_score(y_test_soil, y_pred_soil)
    
    temp_rmse = math.sqrt(mean_squared_error(y_test_temp, y_pred_temp))
    temp_mae = mean_absolute_error(y_test_temp, y_pred_temp)
    temp_r2 = r2_score(y_test_temp, y_pred_temp)
    
    # Store evaluation metrics
    eval_metrics = {
        'Metric': ['RMSE', 'MAE', 'R²'],
        'Rainfall': [rain_rmse, rain_mae, rain_r2],
        'Soil_Moisture': [soil_rmse, soil_mae, soil_r2],
        'Temperature': [temp_rmse, temp_mae, temp_r2]
    }
    
    eval_df = pd.DataFrame(eval_metrics)
    eval_df.to_csv(f"tuned_evaluations/{region}_evaluation.csv", index=False)
    
    # Generate forecasts
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    future_predictions = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_sequence)[0]
        future_predictions.append(pred)
        
        # Get the last sequence without the rolling features
        last_sequence_base = last_sequence[0][:, :3]  # Only take rain, soil, temp
        
        # Create new base sequence
        new_sequence_base = np.vstack([last_sequence_base[1:], pred])
        
        # Calculate rolling statistics for rainfall
        last_rain_values = np.concatenate([last_sequence_base[:, 0], [pred[0]]])
        rolling_mean = np.mean(last_rain_values[-12:])
        rolling_std = np.std(last_rain_values[-12:])
        
        # Create the full sequence with all features
        new_sequence = np.concatenate([
            new_sequence_base,
            np.full((sequence_length, 1), rolling_mean),
            np.full((sequence_length, 1), rolling_std)
        ], axis=1)
        
        # Reshape for the next prediction
        last_sequence = new_sequence.reshape(1, sequence_length, 5)

    # Convert predictions back to original scale
    future_predictions = np.hstack([
        scalers[rain_col].inverse_transform(future_predictions[:, 0].reshape(-1, 1)),
        scalers[soil_col].inverse_transform(future_predictions[:, 1].reshape(-1, 1)),
        scalers[temp_col].inverse_transform(future_predictions[:, 2].reshape(-1, 1))
    ])
    future_dates = pd.date_range(region_data.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq="M")
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        f"{rain_col}_forecast": future_predictions[:, 0],
        f"{soil_col}_forecast": future_predictions[:, 1],
        f"{temp_col}_forecast": future_predictions[:, 2]
    })
    
    # Save forecast with evaluation metrics
    with open(f"tuned_forecasts/{region}_forecast.csv", "w") as f:
        f.write(f"# Evaluation Metrics for {region}\n")
        f.write(f"# Rainfall RMSE: {rain_rmse:.4f}, MAE: {rain_mae:.4f}, R²: {rain_r2:.4f}\n")
        f.write(f"# Soil Moisture RMSE: {soil_rmse:.4f}, MAE: {soil_mae:.4f}, R²: {soil_r2:.4f}\n")
        f.write(f"# Temperature RMSE: {temp_rmse:.4f}, MAE: {temp_mae:.4f}, R²: {temp_r2:.4f}\n")
    
    forecast_df.to_csv(f"tuned_forecasts/{region}_forecast.csv", index=False, mode='a')

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Training History - {region}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"tuned_plots/{region}_training_history.png")                                    
    plt.close()

    # Plot results
    test_dates = region_data.index[train_size+sequence_length:train_size+sequence_length+len(y_test)]
    
    # Rainfall plot
    plt.figure(figsize=(14, 6))
    plt.plot(region_data.index[-100:], region_data[rain_col].values[-100:], label='Historical Rainfall', color='blue')
    plt.plot(future_dates, future_predictions[:, 0], label='Forecast Rainfall', color='red', linestyle='--')
    plt.legend()
    plt.title(f'Rainfall Forecast for {region}\nRMSE: {rain_rmse:.4f}, R²: {rain_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_rainfall_forecast.png")
    plt.close()

    # Soil moisture plot
    plt.figure(figsize=(14, 6))
    plt.plot(region_data.index[-100:], region_data[soil_col].values[-100:], label='Historical Soil Moisture', color='green')
    plt.plot(future_dates, future_predictions[:, 1], label='Forecast Soil Moisture', color='orange', linestyle='--')
    plt.legend()
    plt.title(f'Soil Moisture Forecast for {region}\nRMSE: {soil_rmse:.4f}, R²: {soil_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_soil_forecast.png")
    plt.close()
    
    # Temperature plot
    plt.figure(figsize=(14, 6))
    plt.plot(region_data.index[-100:], region_data[temp_col].values[-100:], label='Historical Temperature', color='purple')
    plt.plot(future_dates, future_predictions[:, 2], label='Forecast Temperature', color='brown', linestyle='--')
    plt.legend()
    plt.title(f'Temperature Forecast for {region}\nRMSE: {temp_rmse:.4f}, R²: {temp_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_temperature_forecast.png")
    plt.close()

    # Test set validation plots
    # Rainfall test validation
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_rain, label='Actual Rainfall', color='blue')
    plt.plot(test_dates, y_pred_rain, label='Predicted Rainfall', color='red', linestyle='--')
    plt.legend()
    plt.title(f'Rainfall Test Set Predictions for {region}\nRMSE: {rain_rmse:.4f}, R²: {rain_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_rainfall_test_validation.png")
    plt.close()
    
    # Soil moisture test validation
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_soil, label='Actual Soil Moisture', color='green')
    plt.plot(test_dates, y_pred_soil, label='Predicted Soil Moisture', color='orange', linestyle='--')
    plt.legend()
    plt.title(f'Soil Moisture Test Set Predictions for {region}\nRMSE: {soil_rmse:.4f}, R²: {soil_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_soil_test_validation.png")
    plt.close()
    
    # Temperature test validation
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_temp, label='Actual Temperature', color='purple')
    plt.plot(test_dates, y_pred_temp, label='Predicted Temperature', color='brown', linestyle='--')
    plt.legend()
    plt.title(f'Temperature Test Set Predictions for {region}\nRMSE: {temp_rmse:.4f}, R²: {temp_r2:.4f}')
    plt.savefig(f"tuned_plots/{region}_temperature_test_validation.png")
    plt.close()

print("Tuned model forecasting and evaluation completed.") 