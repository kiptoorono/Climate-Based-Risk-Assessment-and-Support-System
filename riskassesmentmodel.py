import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from datetime import datetime
import json 
class ClimateRiskModel:
    def __init__(self, data_path, output_dir=None):
        """
        Initialize the climate risk model with rainfall classification.
        
        Parameters:
        data_path (str): Path to CSV with historical and forecasted climate data
        output_dir (str, optional): Directory to save model outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.counties = None
        self.models = {
            'drought': None,
            'heat': None,
            'flood': None,
            'rainfall': None  # New model for rainfall categories
        }
        self.feature_cols = ['rain_z', 'temp_z', 'soil_ratio']
        self.risk_levels = ['Low', 'Mild', 'Moderate', 'Severe']
        self.rainfall_levels = ['Low', 'Moderate', 'High']
        self.score_map = {'Low': 10, 'Mild': 30, 'Moderate': 60, 'Severe': 90}
    
    def load_data(self):
        """Load and preprocess the climate data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['is_forecast'] = (self.df['Date'].dt.year >= 2025).astype(int)
        
        rain_cols = [col for col in self.df.columns if col.endswith('_rain')]
        self.counties = [col.replace('_rain', '') for col in rain_cols]
        print(f"Found {len(self.counties)} counties: {self.counties}")
    
    def prepare_features_and_labels(self, historical_data, county):
        """
        Prepare features and labels for training, including rainfall categories.
        
        Parameters:
        historical_data (DataFrame): Historical subset of data
        county (str): County to process
        
        Returns:
        features (list), labels (dict): Features and risk/rainfall labels
        """
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
            
            # Risk labels (simplified rules for demo; use real events if available)
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
            
            # Rainfall categories based on z-score
            if rain_z <= -1.0:
                labels['rainfall'].append("Low")
            elif rain_z >= 1.0:
                labels['rainfall'].append("High")
            else:
                labels['rainfall'].append("Moderate")
        
        return features, labels
    
    def train_models(self):
        """Train Random Forest models for risks and rainfall categories."""
        historical_data = self.df[self.df['is_forecast'] == 0]
        
        all_features = []
        all_labels = {'drought': [], 'heat': [], 'flood': [], 'rainfall': []}
        
        for county in self.counties:
            features, labels = self.prepare_features_and_labels(historical_data, county)
            if features and labels:
                all_features.extend(features)
                for key in all_labels:
                    all_labels[key].extend(labels[key])
        
        if not all_features:
            raise ValueError("No valid training data available.")
        
        print(f"Training models with {len(all_features)} samples...")
        
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
    
    def predict_risks(self):
        """Predict risks and rainfall levels for all counties and periods."""
        results = {
            'metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'data_period': f"{self.df['Date'].min().strftime('%Y-%m')} to {self.df['Date'].max().strftime('%Y-%m')}",
                'counties_analyzed': self.counties
            },
            'county_risks': {}
        }
        
        periods = {
            'current': self.df[self.df['is_forecast'] == 0].iloc[-3:],
            'near_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year == 2025)],
            'mid_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year.isin([2026, 2027]))],
            'long_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year >= 2028)]
        }
        
        for county in self.counties:
            print(f"Predicting risks for {county}...")
            county_risks = {}
            
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            stats = {
                'rain_mean': self.df[self.df['is_forecast'] == 0][rain_col].mean(),
                'rain_std': max(0.001, self.df[self.df['is_forecast'] == 0][rain_col].std()),
                'soil_mean': self.df[self.df['is_forecast'] == 0][soil_col].mean(),
                'temp_mean': self.df[self.df['is_forecast'] == 0][temp_col].mean(),
                'temp_std': max(0.001, self.df[self.df['is_forecast'] == 0][temp_col].std())
            }
            
            for period_name, period_data in periods.items():
                if period_data.empty or not all(col in period_data.columns for col in [rain_col, soil_col, temp_col]):
                    continue
                
                rain_mean = period_data[rain_col].mean()
                temp_mean = period_data[temp_col].mean()
                soil_mean = period_data[soil_col].mean()
                
                rain_z = (rain_mean - stats['rain_mean']) / stats['rain_std']
                temp_z = (temp_mean - stats['temp_mean']) / stats['temp_std']
                soil_ratio = soil_mean / stats['soil_mean']
                
                X_new = [[rain_z, temp_z, soil_ratio]]
                
                county_risks[period_name] = {
                    'rainfall_amount': rain_mean,
                    'rainfall_level': self.models['rainfall'].predict(X_new)[0],
                    'rainfall_probabilities': dict(zip(self.rainfall_levels, self.models['rainfall'].predict_proba(X_new)[0]))
                }
                
                for risk_type in ['drought', 'heat', 'flood']:
                    if self.models[risk_type]:
                        risk_level = self.models[risk_type].predict(X_new)[0]
                        county_risks[period_name][f'{risk_type}_risk_level'] = risk_level
                        county_risks[period_name][f'{risk_type}_risk_score'] = self.score_map[risk_level]
                        county_risks[period_name][f'{risk_type}_probabilities'] = dict(zip(self.risk_levels, self.models[risk_type].predict_proba(X_new)[0]))
                
                county_risks[period_name].update({
                    'start_date': period_data['Date'].min().strftime('%Y-%m'),
                    'end_date': period_data['Date'].max().strftime('%Y-%m')
                })
            
            results['county_risks'][county] = county_risks
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, 'ml_risk_predictions.json'), 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Predictions saved to {self.output_dir}/ml_risk_predictions.json")
        
        return results

    def run(self):
        """Execute the full risk prediction pipeline."""
        self.load_data()
        self.train_models()
        results = self.predict_risks()
        return results

def main():
    data_path = r"E:\\Agriculture project\\Data\\Processed\\Forecast_Merged_data.csv"
    output_dir = "ml_risk_assessment_results"
    
    try:
        model = ClimateRiskModel(data_path, output_dir)
        results = model.run()
        print("\nML Risk Assessment Summary:")
        print(f"- Analyzed {len(results['county_risks'])} counties")
        print(f"- Results saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()