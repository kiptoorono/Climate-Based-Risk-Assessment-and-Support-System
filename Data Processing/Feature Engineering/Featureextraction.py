import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import os
from datetime import datetime

def extract_climate_features(df, output_path=None):
    """
    Extract climate features for risk assessment and extreme weather forecasting
    
    Parameters:
    df (pandas.DataFrame): DataFrame with climate data including Date and measurements for multiple counties
    output_path (str, optional): Path to save the output features. If None, doesn't save to file.
    
    Returns:
    pandas.DataFrame: DataFrame with extracted features
    """
    print("Starting feature extraction process...")
    
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Ensure Date is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"Processing climate data from {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}")
    
    #----- TEMPORAL FEATURES -----#
    print("Extracting temporal features...")
    
    # Basic time components
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    
    # Kenya's climate seasons 

    conditions = [
        (data['Month'].isin([3, 4, 5])),
        (data['Month'].isin([6, 7, 8, 9])),
        (data['Month'].isin([10, 11, 12])),
        (data['Month'].isin([1, 2]))
    ]
    season_values = ['long_rainy', 'long_dry', 'short_rainy', 'short_dry']
    data['Season'] = np.select(conditions, season_values)
    
    # Binary value wet/dry season indicator
    data['is_wet_season'] = data['Season'].isin(['long_rainy', 'short_rainy']).astype(int)
    
    # Create county lists for each variable type
    rain_cols = [col for col in df.columns if '_rain' in col]
    soil_cols = [col for col in df.columns if '_soil' in col]
    temp_cols = [col for col in df.columns if '_temp' in col]
    counties = [col.split('_')[0] for col in rain_cols]
    
    print(f"Identified {len(counties)} counties and {len(rain_cols + soil_cols + temp_cols)} climate variables")
    
    #----- LAG FEATURES -----#
    print("Calculating lag features...")
    
    # Sort by date to ensure correct lag calculations
    data = data.sort_values('Date')
    
    # Previous month's values(climate of previous month)
    for col in rain_cols + soil_cols + temp_cols:
        data[f'{col}_prev_month'] = data[col].shift(1)
    
    # Same month last year (12-month lag)
    for col in rain_cols + soil_cols + temp_cols:
        data[f'{col}_prev_year'] = data[col].shift(12)
    
    # 3-month rolling averages
    for col in rain_cols + soil_cols + temp_cols:
        data[f'{col}_3month_avg'] = data[col].rolling(window=3, min_periods=1).mean()
    
    #----- SPATIAL FEATURES -----#
    print("Creating spatial features and regional groupings...")
    
    # Group counties into climate zones
    # create geographic clusters using temperature and rainfall patterns
    county_features = pd.DataFrame(index=counties)
    
    for county in counties:
        county_features.loc[county, 'mean_rain'] = data[f'{county}_rain'].mean()
        county_features.loc[county, 'mean_temp'] = data[f'{county}_temp'].mean()
    
    # Cluster counties into 4 climate zones
    kmeans = KMeans(n_clusters=4, random_state=42)
    county_features['zone'] = kmeans.fit_predict(county_features[['mean_rain', 'mean_temp']])
    
    # Map counties to their zones
    county_to_zone = county_features['zone'].to_dict()
    
    # Create zone mapping for each county
    for county in counties:
        data[f'{county}_zone'] = county_to_zone[county]
    
    # Calculate regional averages and differences
    for var_type, cols in zip(['rain', 'soil', 'temp'], [rain_cols, soil_cols, temp_cols]):
        # Create zone averages
        for zone in range(4):
            zone_counties = [county for county in counties if county_to_zone[county] == zone]
            zone_cols = [f'{county}_{var_type}' for county in zone_counties]
            if zone_cols:  # Check if list is not empty
                data[f'zone_{zone}_{var_type}_avg'] = data[zone_cols].mean(axis=1)
            else:
                data[f'zone_{zone}_{var_type}_avg'] = np.nan
        
        # Calculate difference from zone average for each county
        for county in counties:
            zone = county_to_zone[county]
            data[f'{county}_{var_type}_zone_diff'] = data[f'{county}_{var_type}'] - data[f'zone_{zone}_{var_type}_avg']
    
    #----- ANOMALY INDICATORS -----#
    print("Calculating anomaly indicators and extreme event flags...")
    
    # Calculate historical monthly averages for baseline
    historical_means = {}
    historical_stds = {}
    historical_90th = {}
    
    for col in rain_cols + soil_cols + temp_cols:
        # Group by month and calculate statistics
        monthly_stats = data.groupby('Month')[col].agg(['mean', 'std', lambda x: np.percentile(x, 90)])
        historical_means[col] = monthly_stats['mean'].to_dict()
        historical_stds[col] = monthly_stats['std'].to_dict()
        historical_90th[col] = monthly_stats['<lambda_0>'].to_dict()
    
    # Calculate anomalies and extremes
    for col in rain_cols + soil_cols + temp_cols:
        # Monthly value compared to historical average
        data[f'{col}_anomaly'] = data.apply(
            lambda row: row[col] - historical_means[col][row['Month']], axis=1
        )
        
        # Z-scores (standardized anomalies)
        data[f'{col}_zscore'] = data.apply(
            lambda row: (row[col] - historical_means[col][row['Month']]) / 
                       (historical_stds[col][row['Month']] if historical_stds[col][row['Month']] > 0 else 1),
            axis=1
        )
        
        # Binary flags for extreme events (exceeding 90th percentile)
        data[f'{col}_extreme'] = data.apply(
            lambda row: 1 if row[col] > historical_90th[col][row['Month']] else 0,
            axis=1
        )
    
    #----- COMBINED INDICATORS -----#
    print("Creating combined indicators like drought index and heat stress...")
    
    # Drought index (simplified, using standardized values)
    for county in counties:
        # Simplified drought index: combines low rainfall and soil moisture
        rain_zscore = data[f'{county}_rain_zscore']
        soil_zscore = data[f'{county}_soil_zscore']
        
        # Negative values indicate drought conditions
        data[f'{county}_drought_index'] = -(rain_zscore + soil_zscore) / 2
        
        # Drought severity categories
        conditions = [
            (data[f'{county}_drought_index'] < 0),
            (data[f'{county}_drought_index'] >= 0) & (data[f'{county}_drought_index'] < 1),
            (data[f'{county}_drought_index'] >= 1) & (data[f'{county}_drought_index'] < 1.5),
            (data[f'{county}_drought_index'] >= 1.5) & (data[f'{county}_drought_index'] < 2),
            (data[f'{county}_drought_index'] >= 2)
        ]
        values = [0, 1, 2, 3, 4]  # 0=normal, 1=mild, 2=moderate, 3=severe, 4=extreme
        data[f'{county}_drought_severity'] = np.select(conditions, values)
        
        # Heat stress index (combining high temperature and low rainfall)
        temp_zscore = data[f'{county}_temp_zscore']
        data[f'{county}_heat_stress'] = temp_zscore - rain_zscore
        
        # Binary heat stress flag
        data[f'{county}_heat_stress_flag'] = (data[f'{county}_heat_stress'] > 1.5).astype(int)
    
    # Consecutive anomalous months
    for col in rain_cols + soil_cols + temp_cols:
        # Create helper column for streak calculation
        data[f'{col}_is_anomalous'] = (abs(data[f'{col}_zscore']) > 1).astype(int)
        
        # Initialize streak counter
        data[f'{col}_consec_anomaly'] = 0
        streak = 0
        
        # Calculate streaks (this is done in a loop because of the consecutive nature)
        for i in range(len(data)):
            if data.iloc[i][f'{col}_is_anomalous'] == 1:
                streak += 1
            else:
                streak = 0
            data.iloc[i, data.columns.get_loc(f'{col}_consec_anomaly')] = streak
        
        # Drop helper column
        data = data.drop(f'{col}_is_anomalous', axis=1)
    
    # Drop unnecessary columns for the feature set
    data = data.drop([
        col for col in data.columns 
        if any(x in col for x in ['_is_anomalous'])
    ], axis=1, errors='ignore')
    
    # Handle any NaN values that might have been introduced
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    print(f"Feature extraction complete. Generated {data.shape[1]} features.")
    
    # Save to file if path is provided
    if output_path is not None:
        # Create directory 
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Add timestamp to filename 
        if '.csv' not in output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{output_path}_{timestamp}.csv"
        
        data.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        
        # Save county zones information
        zone_info_path = output_path.replace('.csv', '_zones.csv')
        county_features.to_csv(zone_info_path)
        print(f"County zone information saved to {zone_info_path}")
    
    return data


if __name__ == "__main__":
    input_path = r"E:\Agriculture project\Data\Processed\Merged_data.xlsx"
    output_path = r"E:\Agriculture project\Data\Processed\Merged_data_features.csv"
    
    try:
        print(f"Loading data from {input_path}...")
        df = pd.read_excel(input_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Extract features and save to file
        features_df = extract_climate_features(df, output_path=output_path)
        
        print("\nFeature extraction summary:")
        print(f"- Original data shape: {df.shape}")
        print(f"- Feature data shape: {features_df.shape}")
        print(f"- Added {features_df.shape[1] - df.shape[1]} new features")
        print(f"\nAll features saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()