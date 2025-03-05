import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load dataset
file_path = "E:/Agriculture project/rainfall_analysis.xlsx"
df = pd.read_excel(file_path, sheet_name="PNR", parse_dates=["Date"], index_col="Date")

# Filter data from 2000 to 2015
df = df[(df.index.year >= 2015) & (df.index.year <= 2024)]

# Select counties
selected_counties = ["Nyeri", "Meru", "Kisii", "Murang'a", "Kericho"]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, county in enumerate(selected_counties):
    if county in df.columns:
        ax = axes[i]
        ax.plot(df.index, df[county], label=county, color='blue', linewidth=1)
        ax.axhline(75, color='red', linestyle='--', label='Below Normal (75%)')
        ax.axhline(125, color='green', linestyle='--', label='Above Normal (125%)')
        ax.axhline(250, color='purple', linestyle='--', label='High Risk Flooding (250%)')  # Flood threshold
        
        ax.set_title(f"{county} PNR Trend")
        ax.set_ylabel("PNR (%)")
        
        # Reduce number of x-ticks and format them
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Move the legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Ensure all subplots have x-axis labels
for ax in axes:
    plt.setp(ax.get_xticklabels(), visible=True)

# Remove empty subplot (if needed)
if len(selected_counties) < len(axes):
    fig.delaxes(axes[-1])

# Set X-axis label for the entire figure
fig.text(0.5, 0.04, 'Year', ha='center')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
