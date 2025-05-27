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

![Map of the project focus region](map_Screenshot .png)

The system primarily focuses on assessing climate risks within specific counties and agro-ecological zones in Kenya, as visually represented in the map above, providing detailed analysis and recommendations tailored to these regions.

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

```
├── Analysis/                    # Analysis scripts and notebooks
├── Climate risk support/        # Flask web application components
├── Data/                       # Processed and raw data
├── Data Downloading Scripts/    # Scripts for data collection
├── Data Processing/            # Data preprocessing and cleaning
├── LSTM/                       # LSTM-based forecasting models
├── Visualisation/              # Visualization tools and scripts
├── risk_assesment.py          # Core risk assessment module
├── riskassesmentmodel.py      # Risk assessment model implementation
├── Merge Forcasts.py          # Forecast merging and processing
└── requirements.txt           # Project dependencies
```

## Data Sources

### TAMSAT Rainfall Data

The TAMSAT Rainfall Dataset, version v3.1 released on 1st July 2020, provides rainfall estimates and rainfall anomaly estimates for the entire African continent, including Madagascar. This data has a spatial resolution of 0.0375° (approximately 4km) and covers the period from 1st January 1983 to the present. The data is available in netCDF format and includes two main variables: `rfe` for raw rainfall estimates and `rfe_filled` for a temporally complete rainfall record.

### TAMSAT Soil Moisture Data

The TAMSAT Soil Moisture Dataset, version v2.3.1 released in January 2025, offers soil moisture estimates and anomaly estimates relative to the 2001-2020 climatology for the African continent, including Madagascar. This dataset has a spatial resolution of 0.25° (approximately 25km) and temporal coverage from 1st January 1983 to the present. Provided in netCDF format, the key variable is `sm_c4grass`, which represents the soil moisture availability factor for plants, ranging from 0 to 100.

### Data Access

Both the TAMSAT Rainfall Dataset ([https://research.reading.ac.uk/tamsat/rainfall/](https://research.reading.ac.uk/tamsat/rainfall/)) and the TAMSAT Soil Moisture Dataset ([https://research.reading.ac.uk/tamsat/soil-moisture/](https://research.reading.ac.uk/tamsat/soil-moisture/)) are freely available for operational, research, and commercial use under the terms of the Creative Commons Attribution 4.0 International license (CC BY 4.0).

## Live Demo

The application is currently deployed and available live at the following URL: [https://climate-risk-assessment.up.railway.app/](https://climate-risk-assessment.up.railway.app/)

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
   - Place your climate data in the `Data/`