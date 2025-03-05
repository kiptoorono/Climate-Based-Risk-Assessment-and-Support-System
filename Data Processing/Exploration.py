import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rainfall_df= pd.read_excel(r'E:\Agriculture project\Data\Semi-prepared data\Rainfall_total.xlsx')


# Get the unique years in the dataset
years = rainfall_df['Date'].unique()

# Set up the figure with subplots, dynamically adjusting the number of rows based on number of years
fig, axes = plt.subplots(nrows=len(years), ncols=1, figsize=(10, 20), sharex=True)

# Loop through each year and create a subplot
for i, year in enumerate(years):
    # Filter the data for the current year
    subset_df = rainfall_df[rainfall_df['Date'] == year]
    # Plot the monthly rainfall for this year in the corresponding subplot
    axes[i].plot(subset_df['Month'], subset_df['Rainfall'], marker='o')
    axes[i].set_title(f'Rainfall Trend in {year}')
    axes[i].set_ylabel('Rainfall (mm)')
    axes[i].grid(True)

# Add a common x-axis label for all subplots
plt.xlabel('Month')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
