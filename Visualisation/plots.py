import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("E:/Agriculture project/Data/Semi-prepared data/Rainfall_total.xlsx", parse_dates=["Date"], index_col="Date")
selected_counties = ["Nyeri", "Meru", "Kirinyaga", "Murang'a", "Kericho"]
# # Heatmap for spatial variation of PNR using seaborn
# plt.figure(figsize=(12, 8))
# sns.heatmap(df[selected_counties].transpose(), cmap='coolwarm', cbar_kws={'label': 'PNR (%)'})
# plt.title("Heatmap of PNR Across Counties")
# plt.xlabel("Date")
# plt.ylabel("County")
# plt.xticks(ticks=range(len(df.index)), labels=df.index, rotation=90)
# plt.yticks(ticks=range(len(selected_counties)), labels=selected_counties)
# plt.show()

# Bar Plot - Highest & Lowest PNR in a given month
df_monthly = df.resample("M").mean()
latest_month = df_monthly.iloc[-1]
latest_month_sorted = latest_month.sort_values()

plt.figure(figsize=(12, 6))
plt.bar(latest_month_sorted.index, latest_month_sorted.values, color='skyblue')
plt.axhline(75, color='red', linestyle='--', label='Below Normal')
plt.axhline(125, color='blue', linestyle='--', label='Above Normal')
plt.xticks(rotation=90)
plt.title("PNR Distribution for Latest Available Month")
plt.ylabel("PNR (%)")
plt.legend()
plt.show()
