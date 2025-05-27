# Climate-Based Risk Assessment and Support System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System?style=social)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System?style=social)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/network/members)
[![GitHub issues](https://img.shields.io/github/issues/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/pulls)

A comprehensive system for assessing and managing climate-related risks in agricultural regions, providing data-driven insights and recommendations for farmers and agricultural stakeholders.

## Overview

This system combines historical climate data analysis with machine learning-based forecasting to assess various climate risks including drought, heat stress, and flooding. It provides detailed risk assessments at both county and agro-ecological zone levels, along with actionable recommendations for farmers.

## Features

- **Multi-hazard Risk Assessment**
  - Drought risk evaluation
  - Heat stress analysis
  - Flood risk assessment
  - Combined risk scoring

- **Temporal Analysis**
  - Historical data analysis
  - Near-term forecasting (2025)
  - Mid-term forecasting (2026-2027)
  - Long-term forecasting (2028+)

- **Spatial Analysis**
  - County-level assessments
  - Agro-ecological zone mapping
  - Regional risk patterns

- **Decision Support**
  - Crop-specific recommendations
  - Zone-based advisories
  - Risk mitigation strategies
  - Adaptation planning

## Project Structure

```mermaid
graph LR
    A[Climate-Based Risk Assessment and Support System] --> Directories
    A --> Scripts
    A --> L[requirements.txt]

    subgraph Directories
        B[Analysis]
        C[Climate risk support]
        D[Data]
        E[Data Downloading Scripts]
        F[Data Processing]
        G[LSTM]
        H[Visualisation]
    end

    subgraph Scripts
        I[risk_assesment.py]
        J[riskassesmentmodel.py]
        K[Merge Forcasts.py]
    end

    style A fill:#f9f,stroke:#333,stroke-width:4px
    style Directories fill:#f0f8ff,stroke:#333,stroke-width:2px
    style Scripts fill:#f0fff0,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:1px
    style C fill:#bbf,stroke:#333,stroke-width:1px
    style D fill:#bbf,stroke:#333,stroke-width:1px
    style E fill:#bbf,stroke:#333,stroke-width:1px
    style F fill:#bbf,stroke:#333,stroke-width:1px
    style G fill:#bbf,stroke:#333,stroke-width:1px
    style H fill:#bbf,stroke:#333,stroke-width:1px
    style I fill:#bfb,stroke:#333,stroke-width:1px
    style J fill:#bfb,stroke:#333,stroke-width:1px
    style K fill:#bfb,stroke:#333,stroke-width:1px
    style L fill:#bfb,stroke:#333,stroke-width:1px
```

### Directory Descriptions
- **Analysis/**: Contains analysis scripts and notebooks for data processing and visualization
- **Climate risk support/**: Support system components and utilities
- **Data/**: Storage for processed and raw climate data
- **Data Downloading Scripts/**: Scripts for automated data collection from various sources
- **Data Processing/**: Tools for data preprocessing and cleaning
- **LSTM/**: LSTM-based forecasting models and related utilities
- **Visualisation/**: Visualization tools and scripts for data presentation

### Core Scripts
- **risk_assesment.py**: Core risk assessment module
- **riskassesmentmodel.py**: Risk assessment model implementation
- **Merge Forcasts.py**: Forecast merging and processing utilities

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Climate-Based-Risk-Assessment-and-Support-System
```

2. Create and activate a virtual environment:
```bash
python -m venv new_env
source new_env/bin/activate  # On Windows: new_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Dependencies

- **Data Processing**: pandas, numpy, xarray
- **Machine Learning**: tensorflow, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, folium
- **Geospatial**: geopandas, rasterio, rioxarray
- **Web Framework**: Flask

## Usage

1. **Data Preparation**
   - Place your climate data in the `Data/` directory
   - Ensure data includes rainfall, temperature, and soil moisture measurements

2. **Risk Assessment**
```python
from risk_assesment import generate_climate_risk_assessment

# Generate risk assessment
results = generate_climate_risk_assessment(
    data_path='path/to/merged_data.csv',
    output_dir='risk_assessment_results',
    zone_data_path='path/to/zone_data.csv'
)
```

3. **View Results**
   - Check the `risk_assessment_results/` directory for generated reports
   - Review visualizations in the `Visualisation/` directory

## Risk Assessment Methodology

The system uses a comprehensive approach to risk assessment:

1. **Data Analysis**
   - Historical baseline calculation
   - Statistical analysis of climate variables
   - Trend identification

2. **Risk Calculation**
   - Drought Index: Based on rainfall deviation and soil moisture
   - Heat Stress: Temperature anomalies and extreme events
   - Flood Risk: Rainfall intensity and soil saturation

3. **Risk Scoring**
   - Severity levels: Low, Mild, Moderate, Severe
   - Weighted scoring system
   - Zone-specific thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024 Climate-Based Risk Assessment and Support System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

ronobrian058@gmail.com

## Acknowledgments

- [List any acknowledgments or references]