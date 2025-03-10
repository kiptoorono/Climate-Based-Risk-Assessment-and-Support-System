import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import FloatImage

#  Load Climate Zones Data 
file_path = r"E:\Agriculture project\Data\Processed\Merged_data_features_zones.csv"
df = pd.read_csv(file_path)

# Ensure column names are correct
df.columns = df.columns.str.strip()  # Remove spaces
df["county"] = df["county"].str.strip()

#  Load County Boundaries 
county_shapefile = r"E:\Agriculture project\Data\gadm41_KEN_shp\gadm41_KEN_1.shp"
gdf_counties = gpd.read_file(county_shapefile)

# Ensure correct county name column
gdf_counties["NAME_1"] = gdf_counties["NAME_1"].str.strip()

# 3. Merge Data 
gdf_merged = gdf_counties.merge(df, left_on="NAME_1", right_on="county", how="left")

# Define Map and Styling Function 
m = folium.Map(location=[0.5, 37.5], zoom_start=6)

def get_zone_color(zone):
    colors = {0: "blue", 1: "purple", 2: "orange", 3: "green"}
    return colors.get(zone, "gray")

# 5. Add County Boundaries
folium.GeoJson(
    gdf_merged,
    style_function=lambda feature: {
        "fillColor": get_zone_color(feature["properties"]["zone"]),
        "color": "black",
        "weight": 1.5,
        "fillOpacity": 0.5,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["NAME_1", "zone"],
        aliases=["County:", "Climate Zone:"],
        localize=True,
        sticky=False,
        labels=True,
    ),
).add_to(m) 

#add title to the map
title_html = '''
<div style="
    position: fixed;
    top: 10px; left: 50%; 
    transform: translateX(-50%);
    background-color: white;
    padding: 10px; 
    font-size: 20px;
    font-weight: bold;
    z-index: 9999;
    border: 2px solid grey;
    opacity: 0.9;
">
    Climate Zones Clustering using K-Means
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

#   Add Legend on the Side 
legend_html = '''
<div style="
    position: fixed; 
    bottom: 50px; right: 10px; 
    width: 220px; height: 180px; 
    background-color: white; 
    z-index:9999; font-size:14px;
    padding: 10px; border:2px solid grey; 
    opacity: 0.9;
">
<b>Climate Zones</b><br>
<span style="background:blue; width: 12px; height: 12px; display: inline-block;"></span> Zone 0 - Cool & Moderate Rainfall<br>
<span style="background:purple; width: 12px; height: 12px; display: inline-block;"></span> Zone 1 - High Rainfall & Warm<br>
<span style="background:orange; width: 12px; height: 12px; display: inline-block;"></span> Zone 2 - Warm & Moderate Rainfall<br>
<span style="background:green; width: 12px; height: 12px; display: inline-block;"></span> Zone 3 - High Rainfall & Moderate Temperature<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

m.get_root().html.add_child(folium.Element(legend_html))

# Save and Display Map 
m.save("climate_zones_map.html")
print("Map saved as climate_zones_map.html")
