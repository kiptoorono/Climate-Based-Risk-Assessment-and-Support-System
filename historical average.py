import pandas as pd

# Load data
df = pd.read_csv("E:\Agriculture project\Data\Processed\Merged_data_features.csv")

# Keep only columns ending with _rain, _soil, _temp
valid_suffixes = ['_rain', '_soil', '_temp']
climate_cols = [col for col in df.columns if any(col.endswith(suffix) for suffix in valid_suffixes)]

# Subset the DataFrame
df_climate = df[climate_cols]

# Calculate mean for each selected column
historic_means = df_climate.mean().to_dict()

# Convert to DataFrame for easier export/visualization
average_df = pd.DataFrame.from_dict(historic_means, orient='index', columns=["Historic_Average"])
average_df.index.name = "Region_Variable"
average_df.reset_index(inplace=True)

# Save result to CSV
average_df.to_csv("historic_averages_filtered.csv", index=False)

print("✅ Historic averages saved to 'historic_averages_filtered.csv'")
