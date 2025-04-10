import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ClimateRiskAssessment:
    def __init__(self, data_path, output_dir=None, zone_data_path=None):
        """
        Initialize the climate risk assessment system with both ML and traditional approaches.
        
        Parameters:
        data_path (str): Path to CSV with historical and forecasted climate data
        output_dir (str, optional): Directory to save outputs
        zone_data_path (str, optional): Path to CSV file with county zone information
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.zone_data_path = zone_data_path
        self.df = None
        self.counties = None
        self.models = {
            'drought': None,
            'heat': None,
            'flood': None,
            'rainfall': None
        }
        self.feature_cols = ['rain_z', 'temp_z', 'soil_ratio']
        self.risk_levels = ['Low', 'Mild', 'Moderate', 'Severe']
        self.rainfall_levels = ['Low', 'Moderate', 'High']
        self.score_map = {'Low': 10, 'Mild': 30, 'Moderate': 60, 'Severe': 90}
        self.risk_weights = {'drought': 0.33, 'heat': 0.33, 'flood': 0.33}
        self.risk_thresholds = {
            'drought': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
            'heat': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
            'flood': {'severe': 2.0, 'moderate': 1.5, 'mild': 1.0}
        }
        self.crop_zone_mapping = {
            0: ['maize', 'beans', 'potatoes', 'sweet potatoes', 'bananas', 'coffee', 'tea'],
            1: ['maize', 'beans', 'wheat', 'barley', 'potatoes', 'peas', 'dairy'],
            2: ['maize', 'beans', 'sorghum', 'millet', 'cotton', 'sunflower', 'tobacco'],
            3: ['sorghum', 'millet', 'cowpeas', 'green grams', 'cassava', 'livestock']
        }
    
    def load_data(self, encoding='latin1'):
        """Load and preprocess the climate data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, encoding=encoding)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['is_forecast'] = (self.df['Date'].dt.year >= 2025).astype(int)
        
        rain_cols = [col for col in self.df.columns if col.endswith('_rain')]
        self.counties = [col.replace('_rain', '') for col in rain_cols]
        print(f"Found {len(self.counties)} counties: {self.counties}")
        
        # Load zone data
        self.county_zone_mapping = {}
        if self.zone_data_path:
            print(f"Loading zone data from {self.zone_data_path}...")
            zone_df = pd.read_csv(self.zone_data_path, encoding=encoding)
            for _, row in zone_df.iterrows():
                self.county_zone_mapping[row['county']] = int(row['zone'])
            print(f"Loaded zone information for {len(self.county_zone_mapping)} counties")
        else:
            print("Warning: No zone data provided. Counties assigned to default zone 1.")
    
    def prepare_ml_features_and_labels(self, historical_data, county):
        """Prepare features and labels for ML training."""
        rain_col = f'{county}_rain'
        soil_col = f'{county}_soil'
        temp_col = f'{county}_temp'
        
        if not all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
            print(f"Warning: Missing data for {county}, skipping...")
            return None, None
        
        stats = {
            'rain_mean': historical_data[rain_col].mean(),
            'rain_std': max(0.001, historical_data[rain_col].std()),
            'soil_mean': historical_data[soil_col].mean(),
            'temp_mean': historical_data[temp_col].mean(),
            'temp_std': max(0.001, historical_data[temp_col].std())
        }
        
        features = []
        labels = {'drought': [], 'heat': [], 'flood': [], 'rainfall': []}
        
        for _, row in historical_data.iterrows():
            rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
            temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
            soil_ratio = row[soil_col] / stats['soil_mean']
            
            features.append([rain_z, temp_z, soil_ratio])
            
            drought_idx = max(0, -rain_z) * min(2.0, max(0.5, 1 + (1 - soil_ratio)))
            heat_idx = max(0, temp_z)
            flood_idx = rain_z + (0.5 * max(0, soil_ratio - 1))
            
            for risk_type, idx in [('drought', drought_idx), ('heat', heat_idx), ('flood', flood_idx)]:
                if idx >= 2.0:
                    labels[risk_type].append("Severe")
                elif idx >= 1.0:
                    labels[risk_type].append("Moderate")
                elif idx >= 0.5:
                    labels[risk_type].append("Mild")
                else:
                    labels[risk_type].append("Low")
            
            if rain_z <= -1.0:
                labels['rainfall'].append("Low")
            elif rain_z >= 1.0:
                labels['rainfall'].append("High")
            else:
                labels['rainfall'].append("Moderate")
        
        return features, labels
    
    def train_ml_models(self):
        """Train Random Forest models for risks and rainfall categories."""
        historical_data = self.df[self.df['is_forecast'] == 0]
        
        all_features = []
        all_labels = {'drought': [], 'heat': [], 'flood': [], 'rainfall': []}
        
        for county in self.counties:
            features, labels = self.prepare_ml_features_and_labels(historical_data, county)
            if features and labels:
                all_features.extend(features)
                for key in all_labels:
                    all_labels[key].extend(labels[key])
        
        if not all_features:
            raise ValueError("No valid training data available.")
        
        print(f"Training ML models with {len(all_features)} samples...")
        
        for risk_type in ['drought', 'heat', 'flood', 'rainfall']:
            X_train, X_test, y_train, y_test = train_test_split(
                all_features, all_labels[risk_type], test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{risk_type.capitalize()} Model Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred, target_names=self.risk_levels if risk_type != 'rainfall' else self.rainfall_levels))
            self.models[risk_type] = model
    
    def generate_traditional_assessment(self):
        """Generate traditional climate risk assessment with time series data."""
        print("Analyzing current risk conditions...")
        current_period = self.df[self.df['is_forecast'] == 0].iloc[-3:]
        forecast_periods = {
            'near_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year == 2025)],
            'mid_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year.isin([2026, 2027]))],
            'long_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year >= 2028)]
        }
        
        historical_data = self.df[self.df['is_forecast'] == 0]
        county_stats = {}
        print("Calculating historical reference statistics...")
        for county in self.counties:
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            if all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
                county_stats[county] = {
                    'rain_mean': historical_data[rain_col].mean(),
                    'rain_std': max(0.001, historical_data[rain_col].std()),
                    'rain_p10': historical_data[rain_col].quantile(0.10),
                    'soil_mean': historical_data[soil_col].mean(),
                    'soil_std': max(0.001, historical_data[soil_col].std()),
                    'temp_mean': historical_data[temp_col].mean(),
                    'temp_std': max(0.001, historical_data[temp_col].std()),
                    'temp_p90': historical_data[temp_col].quantile(0.90)
                }
        
        risk_results = {
            'metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'data_period': f"{self.df['Date'].min().strftime('%Y-%m')} to {self.df['Date'].max().strftime('%Y-%m')}",
                'counties_analyzed': self.counties,
                'risk_weights': self.risk_weights
            },
            'county_risks': {},
            'zone_risks': {},
            'recommendations': {},
            'crop_advisories': {}
        }
        
        for county in self.counties:
            print(f"Assessing traditional risks for {county}...")
            if county not in county_stats:
                print(f"Warning: No historical reference data for {county}, skipping...")
                continue
            
            zone = self.county_zone_mapping.get(county, 1)
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

                # Initialize time series data
                time_series = {
                    'dates': period_data['Date'].dt.strftime('%Y-%m').tolist(),
                    'drought_risk': [],
                    'heat_risk': [],
                    'flood_risk': []
                }

                # Calculate risks for each time point
                stats = county_stats[county]
                for _, row in period_data.iterrows():
                    rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
                    temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
                    soil_moisture = row[soil_col]
                    
                    # Calculate risk indices
                    drought_idx = max(0, -rain_z) * min(2.0, max(0.5, 1 + (stats['soil_mean'] - soil_moisture) / stats['soil_mean']))
                    heat_idx = max(0, temp_z)
                    flood_idx = rain_z + (0.5 * max(0, soil_moisture / stats['soil_mean'] - 1))
                    
                    # Convert indices to normalized risk scores (0-1)
                    time_series['drought_risk'].append(min(1.0, drought_idx / 2.0))
                    time_series['heat_risk'].append(min(1.0, heat_idx / 2.0))
                    time_series['flood_risk'].append(min(1.0, flood_idx / 2.0))
                
                # Calculate period averages
                rain_mean = period_data[rain_col].mean()
                rain_min = period_data[rain_col].min()
                soil_moisture = period_data[soil_col].mean()
                temp_mean = period_data[temp_col].mean()
                temp_max = period_data[temp_col].max()
                
                rain_zscore = (rain_mean - stats['rain_mean']) / stats['rain_std']
                temp_zscore = (temp_mean - stats['temp_mean']) / stats['temp_std']
                
                drought_index = max(0, -((rain_min - stats['rain_mean']) / stats['rain_std']))
                soil_factor = min(2.0, max(0.5, 1 + (stats['soil_mean'] - soil_moisture) / stats['soil_mean']))
                drought_severity = drought_index * soil_factor if rain_min < stats['rain_mean'] else max(0, 1 - soil_moisture / stats['soil_mean'])
                
                heat_stress = max(0, (temp_max - stats['temp_mean']) / stats['temp_std'])
                
                soil_saturation = soil_moisture / stats['soil_mean']
                flood_index = rain_zscore + (0.5 * max(0, soil_saturation - 1))
                
                for risk_type, index, thresholds in [
                    ('drought', drought_severity, self.risk_thresholds['drought']),
                    ('heat', heat_stress, self.risk_thresholds['heat']),
                    ('flood', flood_index, self.risk_thresholds['flood'])
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
                
                overall_risk_score = (
                    county_risks[period_name]['drought_risk_score'] * self.risk_weights['drought'] +
                    county_risks[period_name]['heat_risk_score'] * self.risk_weights['heat'] +
                    county_risks[period_name]['flood_risk_score'] * self.risk_weights['flood']
                )
                
                county_risks[period_name].update({
                    'time_series': time_series,
                    'drought_index': drought_index,
                    'drought_severity': drought_severity,
                    'heat_stress': heat_stress,
                    'rain_zscore': rain_zscore,
                    'temp_zscore': temp_zscore,
                    'flood_index': flood_index,
                    'soil_moisture': soil_moisture,
                    'soil_mean': stats['soil_mean'],
                    'overall_risk_score': overall_risk_score,
                    'start_date': period_data['Date'].min().strftime('%Y-%m'),
                    'end_date': period_data['Date'].max().strftime('%Y-%m')
                })
            
            risk_results['county_risks'][county] = county_risks
            
            # Generate recommendations
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
                
                suitable_crops = self.crop_zone_mapping.get(int(zone), [])
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
        
        return risk_results
    
    def predict_ml_risks(self):
        """Predict risks using ML models."""
        print("\nGenerating ML-based predictions...")
        results = {
            'metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'counties_analyzed': self.counties
            },
            'county_risks': {}
        }

        # Get historical statistics for each county
        historical_data = self.df[self.df['is_forecast'] == 0]
        county_stats = {}
        for county in self.counties:
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            if all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
                county_stats[county] = {
                    'rain_mean': historical_data[rain_col].mean(),
                    'rain_std': max(0.001, historical_data[rain_col].std()),
                    'soil_mean': historical_data[soil_col].mean(),
                    'soil_std': max(0.001, historical_data[soil_col].std()),
                    'temp_mean': historical_data[temp_col].mean(),
                    'temp_std': max(0.001, historical_data[temp_col].std())
                }

        # Define time periods
        current_period = self.df[self.df['is_forecast'] == 0].iloc[-3:]
        forecast_periods = {
            'near_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year == 2025)],
            'mid_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year.isin([2026, 2027]))],
            'long_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year >= 2028)]
        }

        for county in self.counties:
            print(f"\nProcessing ML predictions for {county}...")
            if county not in county_stats:
                print(f"Warning: No historical reference data for {county}, skipping...")
                continue

            county_risks = {}
            stats = county_stats[county]
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'

            for period_name, period_data in [('current', current_period)] + list(forecast_periods.items()):
                if period_data.empty:
                    continue

                if not all(col in period_data.columns for col in [rain_col, soil_col, temp_col]):
                    continue

                # Calculate time series data for the period
                time_series = {
                    'dates': period_data['Date'].dt.strftime('%Y-%m').tolist(),
                    'drought_risk': [],
                    'heat_risk': [],
                    'flood_risk': []
                }

                # Process each time point in the period
                for _, row in period_data.iterrows():
                    rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
                    temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
                    soil_ratio = row[soil_col] / stats['soil_mean']
                    
                    X_point = [[rain_z, temp_z, soil_ratio]]
                    
                    # Get probabilities for each risk type
                    drought_probs = self.models['drought'].predict_proba(X_point)[0]
                    heat_probs = self.models['heat'].predict_proba(X_point)[0]
                    flood_probs = self.models['flood'].predict_proba(X_point)[0]
                    
                    # Calculate risk scores for time series
                    time_series['drought_risk'].append(sum(p * s for p, s in zip(drought_probs, [0.1, 0.3, 0.6, 0.9])))
                    time_series['heat_risk'].append(sum(p * s for p, s in zip(heat_probs, [0.1, 0.3, 0.6, 0.9])))
                    time_series['flood_risk'].append(sum(p * s for p, s in zip(flood_probs, [0.1, 0.3, 0.6, 0.9])))

                # Calculate period averages for ML predictions
                rain_mean = period_data[rain_col].mean()
                temp_mean = period_data[temp_col].mean()
                soil_mean = period_data[soil_col].mean()
                
                rain_z = (rain_mean - stats['rain_mean']) / stats['rain_std']
                temp_z = (temp_mean - stats['temp_mean']) / stats['temp_std']
                soil_ratio = soil_mean / stats['soil_mean']
                
                X_new = [[rain_z, temp_z, soil_ratio]]
                
                # Get predictions and probabilities
                county_risks[period_name] = {
                    'time_series': time_series,
                    'rainfall_amount': rain_mean,
                    'rainfall_level': self.models['rainfall'].predict(X_new)[0],
                    'rainfall_probabilities': dict(zip(
                        self.rainfall_levels,
                        self.models['rainfall'].predict_proba(X_new)[0].tolist()
                    ))
                }
                
                for risk_type in ['drought', 'heat', 'flood']:
                    if self.models[risk_type]:
                        probabilities = self.models[risk_type].predict_proba(X_new)[0].tolist()
                        risk_level = self.models[risk_type].predict(X_new)[0]
                        
                        county_risks[period_name][f'{risk_type}_risk_level'] = risk_level
                        county_risks[period_name][f'{risk_type}_risk_score'] = self.score_map[risk_level]
                        county_risks[period_name][f'{risk_type}_probabilities'] = dict(zip(
                            self.risk_levels,
                            probabilities
                        ))
                
                county_risks[period_name].update({
                    'start_date': period_data['Date'].min().strftime('%Y-%m'),
                    'end_date': period_data['Date'].max().strftime('%Y-%m')
                })
            
            results['county_risks'][county] = county_risks
        
        return results
    
    def create_risk_visualizations(self, risk_results):
        """Create visualizations of risk assessment results."""
        if not self.output_dir:
            return
            
        viz_dir = os.path.join(self.output_dir, 'visualizations')
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
    
    def run_assessment(self):
        """Run the complete risk assessment pipeline."""
        print("\nStarting combined risk assessment...")
        
        # Load and prepare data
        self.load_data()
        
        # Train ML models
        print("\nTraining ML models...")
        self.train_ml_models()
        
        # Generate traditional assessment
        print("\nGenerating traditional risk assessment...")
        traditional_results = self.generate_traditional_assessment()
        
        # Get ML predictions
        print("\nGenerating ML-based predictions...")
        ml_results = self.predict_ml_risks()
        
        # Combine results
        combined_results = {
            'metadata': traditional_results['metadata'],
            'county_risks': {}
        }
        
        for county in self.counties:
            if county in traditional_results['county_risks'] and county in ml_results['county_risks']:
                combined_results['county_risks'][county] = {
                    'traditional': traditional_results['county_risks'][county],
                    'ml_predictions': ml_results['county_risks'][county],
                    'recommendations': traditional_results['recommendations'].get(county, []),
                    'crop_advisories': traditional_results['crop_advisories'].get(county, {})
                }
        
        # Save results if output directory is specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, 'combined_risk_assessment.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
            
            # Create visualizations
            self.create_risk_visualizations(traditional_results)
        
        return combined_results

def main():
    # File paths
    data_path = r"E:\Agriculture project\Data\Processed\Forecast_Merged_data.csv"
    zone_data_path = r"E:\Agriculture project\Data\Processed\Merged_data_features_zones.csv"
    output_dir = r"E:\Agriculture project\risk_assessment_results"
    
    try:
        # Initialize and run assessment
        assessment = ClimateRiskAssessment(data_path, output_dir, zone_data_path)
        results = assessment.run_assessment()
        
        # Print summary
        print("\nRisk Assessment Summary:")
        print(f"- Analyzed {len(results['county_risks'])} counties")
        print(f"- Generated combined traditional and ML-based assessments")
        print(f"- Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 