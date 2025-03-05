import folium
import geopandas as gpd
import pandas as pd

# Load shapefile and PNR data
gdf = gpd.read_file("E:\Agriculture project\Data\gadm41_KEN_shp\gadm41_KEN_2.shp")
df = pd.read_excel("E:/Agriculture project/Data/Semi-prepared data/Rainfall_total.xlsx", parse_dates=["Date"], index_col="Date")

latest_pnr = df[df.index.year == 2024].mean().reset_index()
latest_pnr.columns = ["County", "PNR"]
latest_pnr["County"] = latest_pnr["County"].str.strip()  # Ensure no leading/trailing spaces

# Merge with shapefile
gdf = gdf.merge(latest_pnr, left_on="NAME_2", right_on="County")

# Convert GeoDataFrame to JSON
gdf_simplified = gdf.copy()
gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(tolerance=0.01, preserve_topology=True)
gdf_json = gdf_simplified.to_json()

# Create base map centered on Kenya with an appropriate zoom level to show the entire country
m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)  # Kenya's lat/lon center

# Add choropleth
folium.Choropleth(
    geo_data=gdf_json,
    name="PNR",
    data=latest_pnr,
    columns=["County", "PNR"],
    key_on="feature.properties.NAME_2",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="PNR (%)",
).add_to(m)
# Save the interactive map to an HTML file
m.save("pnr_map.html")
# Display interactive map
m.save("pnr_map.html")
