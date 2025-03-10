import pandas as pd
import numpy as np

Rainfall_data=pd.read_excel(r"E:\Agriculture project\Data\Semi-prepared data\Rainfall_total.xlsx")
Soil_moisture_data=pd.read_excel(r"E:\Agriculture project\Data\Semi-prepared data\Soilmoisture.xlsx")
Temperature_data=pd.read_excel(r"E:\Agriculture project\Data\Semi-prepared data\Predicted_Temperature.xlsx")

# List of selected counties in Agro-Ecological Zones 1-4
selected_counties = [
    "Nyeri", "Meru", "Embu", "Murang'a",
    "Kericho", "Bomet", "Nandi", "Nyandarua", "Kisii",
    "Uasin Gishu", "Bungoma", "Kakamega",
    "Nakuru", "Laikipia", "Kitui", "Machakos", "Makueni", "Tharaka-Nithi",
    "West Pokot", "Narok", "Baringo"
]

# Function to preprocess data: rename Date column, convert to datetime, and filter counties
def preprocess_data(df, name):
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= 1984]  # Remove data before 1984
    df = df[["Date"] + selected_counties]  # Select only relevant counties
    print(f"{name} Data Loaded | Shape: {df.shape} | Date Range: {df['Date'].min()} to {df['Date'].max()}")
    return df

# Preprocess all datasets
Rainfall_data = preprocess_data(Rainfall_data, "Rainfall")
Soil_moisture_data = preprocess_data(Soil_moisture_data, "Soil Moisture")
Temperature_data = preprocess_data(Temperature_data, "Temperature")

# Check common dates before merging
common_dates = (
    set(Rainfall_data["Date"]) & 
    set(Soil_moisture_data["Date"]) & 
    set(Temperature_data["Date"])
)
print(f"Common Dates Count: {len(common_dates)}")

# Merge datasets and ensure Temperature columns get "_temp" suffix
merged_data = Rainfall_data.merge(Soil_moisture_data, on="Date", suffixes=("_rain", "_soil"))
merged_data = merged_data.merge(Temperature_data, on="Date", how="inner", suffixes=("", "_temp"))

# Manually add "_temp" suffix to Temperature columns
for county in selected_counties:
    if county in merged_data.columns:
        merged_data.rename(columns={county: f"{county}_temp"}, inplace=True)

# Save to Excel
output_path = r"E:\Agriculture project\Data\Processed\Merged_data.xlsx"
merged_data.to_excel(output_path, index=False)
print(f"Data Merged Successfully and saved to {output_path}")
