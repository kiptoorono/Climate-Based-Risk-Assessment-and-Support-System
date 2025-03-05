import pandas as pd
import numpy as np

# Define the file path
file_path = r"E:\Agriculture project\POWER_Regional_Monthly_1984_2022.csv"

# Load the CSV while skipping metadata rows
df = pd.read_csv(file_path, skiprows=9)

# Rename columns properly
df.columns = ["PARAMETER", "YEAR", "LAT", "LON", "JAN", "FEB", "MAR", "APR", 
              "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"]

# Drop the redundant 'PARAMETER' column
df = df.drop(columns=["PARAMETER"])

# Convert numeric columns properly
df = df.apply(pd.to_numeric, errors="coerce")

# Replace missing values (-999) with NaN
df.replace(-999, np.nan, inplace=True)

# Display the cleaned dataframe
print(df.head())

# Save the cleaned data 
cleaned_file_path = r"E:\Agriculture project\Cleaned_POWER_Regional_Monthly.csv"
df.to_csv(cleaned_file_path, index=False)
