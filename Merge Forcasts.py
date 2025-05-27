import pandas as pd
import os

#Define the paths to the forecast files and historic data
forecast_folder = r"E:\Agriculture project\LSTM\forecasts"  # Path to your forecasts folder
historic_file = r"E:\Agriculture project\Data\Processed\Merged_data.xlsx"  # Path to your historic data file

#Read the historic dataset (use pd.read_excel for .xlsx files)
historic_df = pd.read_excel(historic_file, engine='openpyxl')

#List of regions based on the forecast files
regions = [
    "Baringo", "Bomet", "Bungoma", "Embu", "Kakamega", "Kericho", "Kisii", "Kitui",
    "Laikipia", "Machakos", "Makueni", "Meru", "Murang'a", "Nakuru", "Nandi", "Narok",
    "Nyandarua", "Nyeri", "Tharaka-Nithi", "Uasin Gishu", "West Pokot"
]

#Ensure the 'Date' column in the historic dataset is in datetime format
historic_df['Date'] = pd.to_datetime(historic_df['Date'])

#Create a list to store all forecast dataframes
forecast_dfs = []

#Loop through each region to read and process the forecast data
for region in regions:
    # Construct the forecast file path
    forecast_file = os.path.join(forecast_folder, f"{region}_forecast.csv")
    
    # Read the forecast file, skipping the first 4 rows, and specify the encoding
    forecast_df = pd.read_csv(forecast_file, skiprows=4, encoding='cp1252')
    
    # Rename the forecast columns to match the historic dataset
    forecast_df = forecast_df.rename(columns={
        f"{region}_rain_forecast": f"{region}_rain",
        f"{region}_soil_forecast": f"{region}_soil",
        f"{region}_temp_forecast": f"{region}_temp"
    })
    
    # Convert the 'Date' column to datetime format
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    
    # Keep only the columns that match the historic dataset
    forecast_df = forecast_df[['Date', f"{region}_rain", f"{region}_soil", f"{region}_temp"]]
    
    # Add this region's forecast to the list
    forecast_dfs.append(forecast_df)

#Merge all forecast dataframes on the 'Date' column
# Start with the first forecast dataframe
combined_forecast_df = forecast_dfs[0]

# Merge the remaining forecast dataframes
for df in forecast_dfs[1:]:
    combined_forecast_df = combined_forecast_df.merge(
        df,
        on='Date',
        how='outer'  # 'outer' ensures all dates are kept
    )

#Append the combined forecast data to the historic dataset
# Since there's no overlap in dates, we can use pd.concat to append the rows
merged_df = pd.concat([historic_df, combined_forecast_df], ignore_index=True)

#Sort the merged dataset by 'Date' to ensure chronological order
merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)

# Save the merged dataset to a new CSV file
merged_df.to_csv(r"E:\Agriculture project\Data\Processed\Forecast_Merged_data.csv", index=False)
print("Merged dataset saved as 'Forecast_Merged_data.csv'")