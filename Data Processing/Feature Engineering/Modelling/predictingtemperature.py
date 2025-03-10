import pandas as pd
from prophet import Prophet


df = pd.read_excel("E:\Agriculture project\Data\Semi-prepared data\Temperature_wide.xlsx")  # Change to your actual file path

# Convert to long format
df_melted = df.melt(id_vars=['DATE'], var_name='County', value_name='Temperature')


df_melted['DATE'] = pd.to_datetime(df_melted['DATE'])

# Create an empty dataframe to store predictions
predictions = pd.DataFrame()

# Get unique counties
counties = df_melted['County'].unique()

# Forecast for each county
for county in counties:
    print(f"Processing {county}...")  # Show progress

    # Subset data for the county
    county_data = df_melted[df_melted['County'] == county].rename(columns={'DATE': 'ds', 'Temperature': 'y'})

    # Train Prophet model
    model = Prophet()
    model.fit(county_data)

    # Create future dates (24 months for 2023-2024)
    future = model.make_future_dataframe(periods=24, freq='MS')

    # Predict
    forecast = model.predict(future)

    # Keep only future predictions (2023-2024)
    forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'DATE', 'yhat': county})

    # Merge predictions
    if predictions.empty:
        predictions = forecast
    else:
        predictions = predictions.merge(forecast, on='DATE', how='outer')

# Save to Excel
output_path = r"E:\Agriculture project\Data Processing\Predicted_Temperature.xlsx"
predictions.to_excel(output_path, index=False)

print("Predicted Temperature Data Saved Successfully!")
