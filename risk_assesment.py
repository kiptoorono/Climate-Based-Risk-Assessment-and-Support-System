import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def generate_climate_risk_assessment(data_path, output_dir=None, zone_data_path=None, risk_weights=None):
    """
    Generate comprehensive climate risk assessment for farmers across agro-ecological zones
    
    Parameters:
    data_path (str): Path to merged CSV file with historical and forecasted climate data
    output_dir (str, optional): Directory to save risk assessment outputs
    zone_data_path (str, optional): Path to CSV file with county zone information
    risk_weights (dict, optional): Custom weights for overall risk score (e.g., {'drought': 0.5, 'heat': 0.3, 'flood': 0.2})
    
    Returns:
    dict: Risk assessment results including county risks, zone risks, and recommendations
    """
    print(f"Loading merged climate data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Analyzing climate data from {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    print(f"Found {df.shape[0]} months of data, including both historical and forecasted values")
    
    df['is_forecast'] = (df['Date'].dt.year >= 2025).astype(int)
    
    # Load zone data
    county_zone_mapping = {}
    if zone_data_path:
        print(f"Loading zone data from {zone_data_path}...")
        zone_df = pd.read_csv(zone_data_path)
        for _, row in zone_df.iterrows():
            county_zone_mapping[row['county']] = int(row['zone'])
        print(f"Loaded zone information for {len(county_zone_mapping)} counties")
    else:
        print("Warning: No zone data provided. Counties assigned to default zone 1.")
    
    rain_cols = [col for col in df.columns if col.endswith('_rain')]
    counties = [col.replace('_rain', '') for col in rain_cols]
    print(f"Analyzing climate risks for {len(counties)} counties")
    
    # Default risk weights if not provided
    if risk_weights is None:
        risk_weights = {'drought': 0.33, 'heat': 0.33, 'flood': 0.33}  # Equal weighting unless specified
    
    risk_results = {
        'metadata': {
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'data_period': f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}",
            'counties_analyzed': counties,
            'risk_weights': risk_weights
        },
        'county_risks': {},
        'zone_risks': {},
        'recommendations': {},
        'crop_advisories': {}
    }
    
    crop_zone_mapping = {
        0: ['maize', 'beans', 'potatoes', 'sweet potatoes', 'bananas', 'coffee', 'tea'],
        1: ['maize', 'beans', 'wheat', 'barley', 'potatoes', 'peas', 'dairy'],
        2: ['maize', 'beans', 'sorghum', 'millet', 'cotton', 'sunflower', 'tobacco'],
        3: ['sorghum', 'millet', 'cowpeas', 'green grams', 'cassava', 'livestock']
    }
    
    risk_thresholds = {
        'drought': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
        'heat': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
        'flood': {'severe': 2.0, 'moderate': 1.5, 'mild': 1.0}
    }
    
    print("Analyzing current risk conditions...")
    current_period = df[df['is_forecast'] == 0].iloc[-3:]
    forecast_periods = {
        'near_term': df[(df['is_forecast'] == 1) & (df['Date'].dt.year == 2025)],
        'mid_term': df[(df['is_forecast'] == 1) & (df['Date'].dt.year.isin([2026, 2027]))],
        'long_term': df[(df['is_forecast'] == 1) & (df['Date'].dt.year >= 2028)]
    }
    
    historical_data = df[df['is_forecast'] == 0]
    county_stats = {}
    print("Calculating historical reference statistics...")
    for county in counties:
        rain_col = f'{county}_rain'
        soil_col = f'{county}_soil'
        temp_col = f'{county}_temp'
        
        if all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
            county_stats[county] = {
                'rain_mean': historical_data[rain_col].mean(),
                'rain_std': max(0.001, historical_data[rain_col].std()),
                'rain_p10': historical_data[rain_col].quantile(0.10),  # 10th percentile for drought
                'soil_mean': historical_data[soil_col].mean(),
                'soil_std': max(0.001, historical_data[soil_col].std()),
                'temp_mean': historical_data[temp_col].mean(),
                'temp_std': max(0.001, historical_data[temp_col].std()),
                'temp_p90': historical_data[temp_col].quantile(0.90)  # 90th percentile for heat
            }
    
    for county in counties:
        print(f"Assessing risks for {county}...")
        if county not in county_stats:
            print(f"Warning: No historical reference data for {county}, skipping...")
            continue
            
        zone = county_zone_mapping.get(county, 1)  # Default to zone 1 if missing
        county_risks = {'zone': int(zone)}
        
        for period_name, period_data in [('current', current_period)] + list(forecast_periods.items()):
            if period_data.empty:
                continue
            
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            if not all(col in period_data.columns for col in [rain_col, soil_col, temp_col]):
                print(f"Warning: Missing required data columns for {county}, skipping {period_name} period")
                continue
            
            # Use mean and extremes to capture variability
            rain_mean = period_data[rain_col].mean()
            rain_min = period_data[rain_col].min()  # For drought
            soil_moisture = period_data[soil_col].mean()
            temp_mean = period_data[temp_col].mean()
            temp_max = period_data[temp_col].max()  # For heat
            
            stats = county_stats[county]
            rain_zscore = (rain_mean - stats['rain_mean']) / stats['rain_std']
            temp_zscore = (temp_mean - stats['temp_mean']) / stats['temp_std']
            
            # Corrected Drought Calculation
            drought_index = max(0, -((rain_min - stats['rain_mean']) / stats['rain_std']))  # Use min rain for extremes
            soil_factor = min(2.0, max(0.5, 1 + (stats['soil_mean'] - soil_moisture) / stats['soil_mean']))  # Cap at 2x
            drought_severity = drought_index * soil_factor if rain_min < stats['rain_mean'] else max(0, 1 - soil_moisture / stats['soil_mean'])
            
            # Corrected Heat Stress
            heat_stress = max(0, (temp_max - stats['temp_mean']) / stats['temp_std'])  # Use max temp
            
            # Corrected Flood Risk: Combine rain and soil moisture
            soil_saturation = soil_moisture / stats['soil_mean']
            flood_index = rain_zscore + (0.5 * max(0, soil_saturation - 1))  # Boost if soil is saturated
            
            print(f"  {period_name} period diagnostics:")
            print(f"    Rain: Mean={rain_mean:.2f}, Min={rain_min:.2f}, Z-score={rain_zscore:.2f}")
            print(f"    Temp: Mean={temp_mean:.2f}, Max={temp_max:.2f}, Z-score={temp_zscore:.2f}")
            print(f"    Soil: {soil_moisture:.2f}, Saturation={soil_saturation:.2f}")
            print(f"    Drought: Index={drought_index:.2f}, Severity={drought_severity:.2f}")
            print(f"    Heat: {heat_stress:.2f}")
            print(f"    Flood: Index={flood_index:.2f}")
            
            # Assign risk levels
            for risk_type, index, thresholds in [
                ('drought', drought_severity, risk_thresholds['drought']),
                ('heat', heat_stress, risk_thresholds['heat']),
                ('flood', flood_index, risk_thresholds['flood'])
            ]:
                if index >= thresholds['severe']:
                    level, score = "Severe", 90
                elif index >= thresholds['moderate']:
                    level, score = "Moderate", 60
                elif index >= thresholds['mild']:
                    level, score = "Mild", 30
                else:
                    level, score = "Low", 10
                county_risks.setdefault(period_name, {})[f'{risk_type}_risk_level'] = level
                county_risks[period_name][f'{risk_type}_risk_score'] = score
            
            # Overall risk score with configurable weights
            overall_risk_score = (
                county_risks[period_name]['drought_risk_score'] * risk_weights['drought'] +
                county_risks[period_name]['heat_risk_score'] * risk_weights['heat'] +
                county_risks[period_name]['flood_risk_score'] * risk_weights['flood']
            )
            
            county_risks[period_name].update({
                'drought_index': drought_index,
                'drought_severity': drought_severity,
                'heat_stress': heat_stress,
                'rain_zscore': rain_zscore,
                'flood_index': flood_index,
                'soil_moisture': soil_moisture,
                'overall_risk_score': overall_risk_score,
                'start_date': period_data['Date'].min().strftime('%Y-%m'),
                'end_date': period_data['Date'].max().strftime('%Y-%m')
            })
        
        risk_results['county_risks'][county] = county_risks
        
        # Recommendations (unchanged logic, just using corrected risks)
        recommendations = []
        risk_period = county_risks.get('near_term', county_risks.get('current', {}))
        if risk_period:
            if risk_period.get('drought_risk_level') == "Severe":
                recommendations.extend(["URGENT: Implement water conservation", "Use drought-resistant crops"])
            elif risk_period.get('drought_risk_level') == "Moderate":
                recommendations.extend(["Monitor water resources", "Prepare irrigation"])
            if risk_period.get('heat_risk_level') == "Severe":
                recommendations.extend(["URGENT: Protect crops from heat", "Increase irrigation"])
            elif risk_period.get('heat_risk_level') == "Moderate":
                recommendations.append("Monitor crops for heat stress")
            if risk_period.get('flood_risk_level') == "Severe":
                recommendations.extend(["URGENT: Clear drainage", "Elevate planting beds"])
            elif risk_period.get('flood_risk_level') == "Moderate":
                recommendations.append("Maintain drainage")
            risk_results['recommendations'][county] = recommendations
            
            suitable_crops = crop_zone_mapping.get(int(zone), [])
            crop_advisories = {}
            for crop in suitable_crops:
                crop_recs = []
                if risk_period.get('drought_risk_level') in ["Moderate", "Severe"] and crop in ['maize', 'beans']:
                    crop_recs.append(f"Use drought-resistant {crop}")
                if risk_period.get('heat_risk_level') in ["Moderate", "Severe"] and crop in ['coffee', 'tea']:
                    crop_recs.append(f"Shade {crop} plants")
                if risk_period.get('flood_risk_level') in ["Moderate", "Severe"] and crop in ['beans', 'potatoes']:
                    crop_recs.append(f"Raise {crop} beds")
                if crop_recs:
                    crop_advisories[crop] = crop_recs
            risk_results['crop_advisories'][county] = crop_advisories
    
    # Zone-level risks (unchanged logic, using corrected county risks)
    print("Calculating zone-level risk aggregates...")
    for zone in range(4):
        zone_counties = [c for c in counties if c in risk_results['county_risks'] and risk_results['county_risks'][c]['zone'] == zone]
        if not zone_counties:
            continue
        zone_risks = {'counties': zone_counties}
        for period in ['current', 'near_term', 'mid_term', 'long_term']:
            counties_with_period = [c for c in zone_counties if period in risk_results['county_risks'][c]]
            if not counties_with_period:
                continue
            zone_risks[period] = {
                'drought_risk': np.mean([risk_results['county_risks'][c][period]['drought_risk_score'] for c in counties_with_period]),
                'heat_risk': np.mean([risk_results['county_risks'][c][period]['heat_risk_score'] for c in counties_with_period]),
                'flood_risk': np.mean([risk_results['county_risks'][c][period]['flood_risk_score'] for c in counties_with_period]),
                'overall_risk': np.mean([risk_results['county_risks'][c][period]['overall_risk_score'] for c in counties_with_period]),
                'start_date': risk_results['county_risks'][counties_with_period[0]][period]['start_date'],
                'end_date': risk_results['county_risks'][counties_with_period[0]][period]['end_date']
            }
        risk_results['zone_risks'][int(zone)] = zone_risks
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'risk_assessment_results.json'), 'w') as f:
            json.dump(risk_results, f, indent=4)
        print(f"\nRisk assessment results saved to {output_dir}")
        create_risk_visualizations(risk_results, output_dir)
    
    return risk_results

def create_risk_visualizations(risk_results, output_dir):
    # Unchanged visualization logic, just using corrected data
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    counties, drought_risks, heat_risks, flood_risks, overall_risks, zones = [], [], [], [], [], []
    for county, risks in risk_results['county_risks'].items():
        if 'current' in risks:
            counties.append(county)
            drought_risks.append(risks['current'].get('drought_risk_score', 0))
            heat_risks.append(risks['current'].get('heat_risk_score', 0))
            flood_risks.append(risks['current'].get('flood_risk_score', 0))
            overall_risks.append(risks['current'].get('overall_risk_score', 0))
            zones.append(f"Zone {risks['zone']}")
    
    if counties:
        risk_df = pd.DataFrame({
            'County': counties, 'Drought Risk': drought_risks, 'Heat Risk': heat_risks,
            'Flood Risk': flood_risks, 'Zone': zones, 'Overall Risk': overall_risks
        }).sort_values(['Zone', 'Overall Risk'], ascending=[True, False])
        
        fig, ax = plt.subplots(figsize=(10, len(counties) * 0.4))
        sns.heatmap(risk_df[['Drought Risk', 'Heat Risk', 'Flood Risk']].values, cmap='YlOrRd', annot=True, fmt='.0f',
                    yticklabels=[f"{c} ({z})" for c, z in zip(risk_df['County'], risk_df['Zone'])], ax=ax)
        ax.set_xticklabels(['Drought Risk', 'Heat Risk', 'Flood Risk'])
        plt.title('Climate Risk Assessment by County')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'county_risk_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    data_path = r"E:\\Agriculture project\\Data\\Processed\\Forecast_Merged_data.csv"
    zone_data_path = r"E:\Agriculture project\Data\Processed\Merged_data_features_zones.csv"
    output_dir = "risk_assessment_results"
    
    try:
        risk_results = generate_climate_risk_assessment(data_path, output_dir, zone_data_path)
        print("\nRisk Assessment Summary:")
        print(f"- Analyzed {len(risk_results['county_risks'])} counties across {len(risk_results['zone_risks'])} zones")
        print(f"- Generated recommendations for {len(risk_results['recommendations'])} counties")
        print(f"- Created crop advisories for {len(risk_results['crop_advisories'])} counties")
        print(f"\nResults saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()