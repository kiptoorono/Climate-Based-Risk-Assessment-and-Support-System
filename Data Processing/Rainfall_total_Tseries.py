# Average ares
import os
import geopandas as gpd
import xarray as xr
import rioxarray
import pandas as pd
from shapely.geometry import mapping

# Function to process all years and create a time series of each county in an excel
def create_rainfall_moisture_time_series(base_dir, shapefile_path, start_year, end_year, excel_path):
    # Load Shapefile
    wards = gpd.read_file(shapefile_path)
    
    # Initialize dictionary to store data
    time_series_data = {ward : [] for ward in wards['NAME_1']} 

    # Iterate through each year
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.isdir(year_dir):
            continue

        print(f"Processing year: {year}")

        # Initialize list to store monthly averages for the selected year
        monthly_averages_selected_year = []

        # Iterate over each month directory within the selected year
        for month_dir in sorted(os.listdir(year_dir), key=lambda x: int(x)):  # Sort month directories as integers
            month_path = os.path.join(year_dir, month_dir)

            # Check if it's a directory
            if not os.path.isdir(month_path):
                continue  # Skip if it's not a directory

            # Create a list of nc file paths within that month
            data_files = [f for f in os.listdir(month_path) if f.endswith('.nc')]

            # Initialize an empty list to store datasets
            datasets = []

            # Loop through each data file, open dataset, and append to datasets list
            for data_file in data_files:
                file_path = os.path.join(month_path, data_file)
                data = xr.open_dataset(file_path)
                datasets.append(data)

            # Concatenate datasets along time dimension
            combined_data = xr.concat(datasets, dim='time')

            # Calculate total of 'rfe_filled' along the time dimension (monthly average)
            sum_rfe_filled_monthly = combined_data['rfe_filled'].sum(dim='time')

            # Set spatial dimensions and CRS
            sum_rfe_filled_monthly.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
            sum_rfe_filled_monthly.rio.write_crs('epsg:4326', inplace=True)

            # Clip data to Kenya boundary using Level 3 shapefile
            try:
                clipped_data_monthly = sum_rfe_filled_monthly.rio.clip(wards.geometry.apply(mapping), wards.crs, drop=True)

                # Append monthly average to list for the selected year
                monthly_averages_selected_year.append((month_dir, clipped_data_monthly))
                
                # Append data for each ward to the time series data
                for ward in wards.itertuples():
                    ward_name = ward.NAME_1

                    # Clip the rainfall data to the specific ward's geometry
                    clipped_ward_data = clipped_data_monthly.rio.clip([mapping(ward.geometry)], wards.crs, drop=True)

                    # Calculate the mean rainfall for the entire ward
                    ward_mean_rainfall = clipped_ward_data.mean().item()

                    # Create a 'Date' in the format 'YYYY-MM-DD' using the Year, Month, and default day '01'
                    date_str = f"{year}-{int(month_dir):02d}-01"  # Ensure proper formatting with zero-padded month
                    time_series_data[ward_name].append({
                        'Date': date_str,
                        'rainfall': ward_mean_rainfall
                    })
            except Exception as e:
                print(f"Error processing {month_dir} {year}: {e}")

    # Create DataFrame to store the time series data
    all_data = []
    for ward_name, values in time_series_data.items():
        for entry in values:
            entry['Ward'] = ward_name
            all_data.append(entry)

    time_series_df = pd.DataFrame(all_data)

    # Ensure the 'Date' column is in datetime format for proper sorting
    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'], format='%Y-%m-%d')

    # Pivot the DataFrame to have wards as columns and time periods as rows
    pivot_df = time_series_df.pivot_table(index='Date', columns='Ward', values='rainfall')

    # Sort DataFrame by the Date column to ensure proper order
    pivot_df.sort_index(inplace=True)

    # Save DataFrame to Excel
    pivot_df.to_excel(excel_path)
    print(f"Time series data saved to {excel_path}")

# Parameters
base_dir = r'E:\Agriculture project\Data\Raw data\Rainfall data'
shapefile_path = r'E:\Agriculture project\gadm41_KEN_shp\gadm41_KEN_1.shp'
start_year = 1983
end_year = 2024
excel_path = r"E:\Agriculture project\Data\Semi-prepared data\Rainfall_total.xlsx"

# Create time series data
create_rainfall_moisture_time_series(base_dir, shapefile_path, start_year, end_year, excel_path)
