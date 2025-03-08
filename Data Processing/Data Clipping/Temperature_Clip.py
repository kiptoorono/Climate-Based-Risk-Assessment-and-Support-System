import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# File paths
shapefile_path = r"E:\Agriculture project\gadm41_KEN_shp\gadm41_KEN_3.shp"  # Level 3 shapefile (wards)
csv_path = r"E:\Agriculture project\Cleaned_POWER_Regional_Monthly.csv"
output_excel_path = r"E:\Agriculture project\Data\Semi-prepared data\Temperature_wide.xlsx"

# Load Kenya boundaries (wards)
kenya_boundaries = gpd.read_file(shapefile_path)

# Load temperature CSV data
df = pd.read_csv(csv_path)

# Melt the dataset to long format (Convert JAN-DEC into a single column)
df_long = df.melt(id_vars=['YEAR', 'LAT', 'LON'], 
                  value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
                  var_name='MONTH', value_name='VALUE')

# Map month names to numbers
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

df_long['MONTH'] = df_long['MONTH'].map(month_map)

# Create a proper `DATE` column
df_long['DATE'] = pd.to_datetime(df_long[['YEAR', 'MONTH']].assign(DAY=1))  # Using the 1st of the month

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df_long.LON, df_long.LAT)]
gdf = gpd.GeoDataFrame(df_long, geometry=geometry, crs="EPSG:4326")

# Perform spatial join with Kenya boundaries
gdf = gpd.sjoin(gdf, kenya_boundaries[['NAME_1', 'geometry']], how="left", predicate="within")

# Check for missing  assignments
missing = gdf[gdf['NAME_1'].isna()]
if not missing.empty:
    print(f"Warning: {len(missing)} records could not be assigned to a .")

# Group by Date & , keeping monthly values
df_grouped = gdf.groupby(['DATE', 'NAME_1']).mean(numeric_only=True).reset_index()

# Pivot to wide format
df_pivot = df_grouped.pivot(index='DATE', columns='NAME_1', values='VALUE')

# Save to Excel
df_pivot.to_excel(output_excel_path)

print(f"Processed temperature data saved to {output_excel_path}")
