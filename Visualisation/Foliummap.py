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
m = folium.Map(
    location=[0.0236, 37.9062],
    zoom_start=6,
    zoom_control=False,  # Disable default zoom control
    tiles='OpenStreetMap'
)

# Add custom zoom control
custom_zoom_control = '''
<div class="leaflet-control-container">
    <div class="leaflet-top leaflet-right">
        <div class="leaflet-control-zoom leaflet-bar leaflet-control" style="margin-top: 20px;">
            <a class="leaflet-control-zoom-in" href="#" title="Zoom in" role="button" aria-label="Zoom in" onclick="map.zoomIn()">+</a>
            <a class="leaflet-control-zoom-out" href="#" title="Zoom out" role="button" aria-label="Zoom out" onclick="map.zoomOut()">-</a>
            <a class="leaflet-control-zoom-reset" href="#" title="Reset view" role="button" aria-label="Reset view" 
               onclick="map.setView([0.0236, 37.9062], 6)" 
               style="background-color: #fff; width: 26px; height: 26px; line-height: 26px; text-align: center; text-decoration: none; color: black; font-weight: bold;">
               R
            </a>
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(custom_zoom_control))

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

# Add Legend with fixed styling
legend_html = '''
<div style="
    position: fixed; 
    bottom: 20px; right: 10px; 
    width: 220px;
    background-color: white; 
    z-index: 1000; 
    font-size: 14px;
    padding: 10px; 
    border: 2px solid grey; 
    border-radius: 4px;
    opacity: 0.9;
">
<b>Climate Zones</b><br>
<div style="margin-top: 8px;">
<span style="background:blue; width: 12px; height: 12px; display: inline-block; margin-right: 6px;"></span> Zone 0 - Cool & Moderate Rainfall<br>
<span style="background:purple; width: 12px; height: 12px; display: inline-block; margin-right: 6px;"></span> Zone 1 - High Rainfall & Warm<br>
<span style="background:orange; width: 12px; height: 12px; display: inline-block; margin-right: 6px;"></span> Zone 2 - Warm & Moderate Rainfall<br>
<span style="background:green; width: 12px; height: 12px; display: inline-block; margin-right: 6px;"></span> Zone 3 - High Rainfall & Moderate Temperature
</div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save Map 
m.save("Climate risk support/static/climate_zones_map.html")
print("Map saved as climate_zones_map.html in static directory")
