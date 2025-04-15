import os
from flask import Flask, render_template, send_from_directory, jsonify, request
import pandas as pd
from datetime import datetime
import json
import logging

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

@app.route('/risk_assessment.html')
def risk_assessment():
    return send_from_directory(app.static_folder, 'risk_assessment.html')

@app.route('/about.html')
def about():
    return send_from_directory(app.static_folder, 'about.html')

@app.route('/api/forecast/<location>')
def get_forecast_data(location):
    try:
        # Convert location name to match file naming
        location = location.lower()  # Keep lowercase for internal use
        location_cap = location.capitalize()  # For file name
        
        # Define path for forecast file
        base_path = r'E:\Agriculture project\LSTM\forecasts'
        forecast_file = os.path.join(base_path, f'{location_cap}_forecast.csv')
        
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

@app.route('/api/risk_assessment/<location>')
def get_risk_assessment(location):
    try:
        # Path to pre-computed risk assessment file
        risk_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'risk_assessment_results', 'combined_risk_assessment.json')
        
        try:
            # Try to read the pre-computed risk assessments
            with open(risk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded risk assessment data")
                
                if 'county_risks' not in data:
                    return jsonify({
                        'error': 'Invalid data structure: missing county_risks'
                    }), 500
                
                risk_data = data['county_risks']
                print(f"Available locations: {list(risk_data.keys())}")  # Debug log
                print(f"Requested location: {location}")  # Debug log
                
                # Try to find a case-insensitive match
                location_found = None
                for key in risk_data.keys():
                    if key.lower() == location.lower():
                        location_found = key
                        break
            
                if location_found:
                    print(f"Found matching location: {location_found}")
                    return jsonify({
                        'location': location_found,
                        'risk_assessment': risk_data[location_found]
                    })
                else:
                    print(f"Location not found. Available locations: {list(risk_data.keys())}")
                    return jsonify({
                        'error': f'No risk assessment data available for {location}. Available locations: {", ".join(sorted(risk_data.keys()))}'
                    }), 404
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading risk assessment file: {str(e)}")
            return jsonify({
                'error': f'Could not read risk assessment data: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error loading risk assessment data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True) 