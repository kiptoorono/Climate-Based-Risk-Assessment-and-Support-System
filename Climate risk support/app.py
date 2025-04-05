from flask import Flask, render_template, send_from_directory, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/forecast.html')
def forecast():
    return send_from_directory(app.static_folder, 'forecast.html')

@app.route('/climate_zones.html')
def climate_zones():
    return send_from_directory(app.static_folder, 'climate_zones.html')

@app.route('/climate_zones_map.html')
def climate_zones_map():
    return send_from_directory(app.static_folder, 'climate_zones_map.html')

@app.route('/api/forecast/<location>')
def get_forecast_data(location):
    try:
        # Convert location name to match file naming
        location = location.lower()  # Convert to lowercase for file name
        location_cap = location.capitalize()  # Capitalized for column names
        
        # Define path for forecast file
        base_path = r'E:\Agriculture project\LSTM\forecasts'
        forecast_file = os.path.join(base_path, f'{location}_forecast.csv')
        
        print(f"Looking for forecast file at: {forecast_file}")
        
        if not os.path.exists(forecast_file):
            return jsonify({
                'success': False,
                'error': f'No forecast data available for {location}'
            })
        
        try:
            # Skip the metrics and read only the data portion
            data_df = pd.read_csv(forecast_file, skiprows=4, encoding='latin-1')
            print("Successfully read data rows")
            print(f"Columns found: {data_df.columns.tolist()}")
            
            # Debug print for data
            print("\nFirst few rows of data:")
            print(data_df.head())
            
            # Replace empty strings and whitespace with NaN
            data_df = data_df.replace(r'^\s*$', pd.NA, regex=True)
            data_df = data_df.replace('', pd.NA)
            
            # Get column names with proper capitalization
            rain_col = f'{location_cap}_rain_forecast'
            soil_col = f'{location_cap}_soil_forecast'
            temp_col = f'{location_cap}_temp_forecast'
            
            # Verify column names exist
            if rain_col not in data_df.columns or soil_col not in data_df.columns or temp_col not in data_df.columns:
                print(f"Expected columns not found. Looking for: {rain_col}, {soil_col}, {temp_col}")
                print(f"Available columns: {data_df.columns.tolist()}")
                return jsonify({
                    'success': False,
                    'error': f'Column names not found in forecast file'
                })
            
            # Debug print before conversion
            print(f"\nUnique values in rainfall column: {data_df[rain_col].unique()}")
            print(f"Unique values in soil moisture column: {data_df[soil_col].unique()}")
            print(f"Unique values in temperature column: {data_df[temp_col].unique()}")
            
            # Convert columns to float, replacing any remaining non-numeric values with NaN
            try:
                data_df[rain_col] = pd.to_numeric(data_df[rain_col], errors='coerce')
                data_df[soil_col] = pd.to_numeric(data_df[soil_col], errors='coerce')
                data_df[temp_col] = pd.to_numeric(data_df[temp_col], errors='coerce')
            except Exception as e:
                print(f"Error converting columns to numeric: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error converting data to numeric: {str(e)}'
                })
            
            # Remove rows with NaN values
            data_df = data_df.dropna()
            
            if len(data_df) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No valid data rows found after cleaning'
                })
            
            # Extract forecast data (without metrics)
            data = {
                'dates': data_df['Date'].tolist(),
                'rainfall': data_df[rain_col].tolist(),
                'soil_moisture': data_df[soil_col].tolist(),
                'temperature': data_df[temp_col].tolist(),
                'metrics': {
                    'rainfall': {'rmse': 0, 'mae': 0, 'r2': 0},
                    'soil_moisture': {'rmse': 0, 'mae': 0, 'r2': 0},
                    'temperature': {'rmse': 0, 'mae': 0, 'r2': 0}
                }
            }
            
            print(f"\nProcessed data for {location}:")
            print(f"Number of dates: {len(data['dates'])}")
            print(f"Number of values per variable: {len(data['rainfall'])}")
            
            return jsonify({
                'success': True,
                'data': data
            })
            
        except Exception as e:
            print(f"Error processing {forecast_file}: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing forecast data: {str(e)}'
            })
            
    except Exception as e:
        print(f"Error processing request for {location}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True) 