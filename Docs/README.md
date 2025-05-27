# Climate-Based Risk Assessment and Support System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System?style=flat&logo=github&logoColor=white)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System?style=flat&logo=github&logoColor=white)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/network/members)
[![GitHub issues](https://img.shields.io/github/issues/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System.svg?style=flat&logo=github&logoColor=white)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System.svg?style=flat&logo=github&logoColor=white)](https://github.com/kiptoorono/Climate-Based-Risk-Assessment-and-Support-System/pulls)

A comprehensive system for assessing and managing climate-related risks in agricultural regions, providing data-driven insights and recommendations for farmers and agricultural stakeholders.

## Overview

This system combines historical climate data analysis with machine learning-based forecasting to assess various climate risks including drought, heat stress, and flooding. It provides detailed risk assessments at both county and agro-ecological zone levels, along with actionable recommendations for farmers.

![Map of the project focus region](map_Screenshot.png)

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
  - Zone-based advisories
  - Risk mitigation strategies
  - Adaptation planning

## Project Structure

```
📂 Analysis/                    -  EDA Results
📂 Climate risk support/        -  Flask web application components (standalone)
📂 Data/                        -  Processed and raw data
📂 Data Downloading Scripts/    -  Web crawler Scripts for data collection
📂 Data Processing/             -  Geo-spatial clipping
📂 Feature Engineering/         -  Engineering semi-prepared data for LSTM 
📂 LSTM/                        -  LSTM-based forecasting model
📂 Visualisation/               -  Visualization tools and scripts
📄 risk_assesment.py            -  Statistical risk assessment module
📄 riskassesmentmodel.py        -  Ml based Risk assessment model 
📄 Merge Forcasts.py            -  Forecast & Historical data merging 
📄 requirements.txt             -  Project dependencies
```

## Execution Workflow

To run the core climate risk assessment process, follow these steps in order:

1.  **Data Downloading:** Execute the scripts in the `Data Downloading Scripts/` directory to collect raw climate data.
2.  **Data Processing:** Run the scripts in the `Data Processing/` directory to clean and preprocess the raw data, preparing it for feature engineering.
3.  **Feature Engineering:** Execute the scripts in the `Feature Engineering/` directory to engineer features required for the LSTM model and risk assessment.
4.  **LSTM Forecasting:** Utilize the scripts/models in the `LSTM/` folder to generate climate forecasts based on the engineered data.
5.  **Risk Assessment:** Run the `risk_assesment.py` script and the models in `riskassesmentmodel.py`, which use the processed data and LSTM forecasts to perform the climate risk assessment.

The `Climate risk support/` directory contains a standalone Flask web application that provides a user interface for interacting with the system, including viewing results and potentially running parts of the workflow. This application can be run independently after the necessary data has been processed and features engineered.

## Data Sources

### TAMSAT Rainfall Data

The TAMSAT Rainfall Dataset, version v3.1 released on 1st July 2020, provides rainfall estimates and rainfall anomaly estimates for the entire African continent, including Madagascar. This data has a spatial resolution of 0.0375° (approximately 4km) and covers the period from 1st January 1983 to the present. The data is available in netCDF format and includes two main variables: `rfe` for raw rainfall estimates and `rfe_filled` for a temporally complete rainfall record.

### TAMSAT Soil Moisture Data

The TAMSAT Soil Moisture Dataset, version v2.3.1 released in January 2025, offers soil moisture estimates and anomaly estimates relative to the 2001-2020 climatology for the African continent, including Madagascar. This dataset has a spatial resolution of 0.25° (approximately 25km) and temporal coverage from 1st January 1983 to the present. Provided in netCDF format, the key variable is `sm_c4grass`, which represents the soil moisture availability factor for plants, ranging from 0 to 100.

### Data Access

Both the TAMSAT Rainfall Dataset ([https://research.reading.ac.uk/tamsat/rainfall/](https://research.reading.ac.uk/tamsat/rainfall/)) and the TAMSAT Soil Moisture Dataset ([https://research.reading.ac.uk/tamsat/soil-moisture/](https://research.reading.ac.uk/tamsat/soil-moisture/)) are freely available for operational, research, and commercial use under the terms of the Creative Commons Attribution 4.0 International license (CC BY 4.0).

## Live Demo

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=flat)](https://climate-risk-assessment.up.railway.app/)

The application is currently deployed and available live at the following URL: [https://climate-risk-assessment.up.railway.app/](https://climate-risk-assessment.up.railway.app/)

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd Climate-Based-Risk-Assessment-and-Support-System
```

2. Create and activate a virtual environment:

   **For Linux/macOS:**
   ```bash
   python3 -m venv new_env
   source new_env/bin/activate
   ```

   **For Windows:**
   ```bash
   python -m venv new_env
   new_env\Scripts\activate
   ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Key Dependencies

- **Data Processing**:
  [![pandas](https://img.shields.io/badge/pandas-blue?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![numpy](https://img.shields.io/badge/numpy-blue?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
  [![xarray](https://img.shields.io/badge/xarray-F26B00?style=flat&logo=xarray&logoColor=white)](http://xarray.pydata.org/)
- **Machine Learning**:
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-0865A5?style=flat&logo=xgboost&logoColor=white)](https://xgboost.ai/)
- **Visualization**:
  [![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=flat&logo=matplotlib&logoColor=black)](https://matplotlib.org/)
  [![Seaborn](https://img.shields.io/badge/Seaborn-377EB9?style=flat&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
  [![Folium](https://img.shields.io/badge/Folium-0078A8?style=flat&logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
- **Geospatial**:
  [![GeoPandas](https://img.shields.io/badge/GeoPandas-006064?style=flat&logo=conda-forge&logoColor=white)](https://geopandas.org/)
  [![Rasterio](https://img.shields.io/badge/Rasterio-00796B?style=flat&logo=conda-forge&logoColor=white)](https://rasterio.readthedocs.io/en/latest/)
  [![Rioxarray](https://img.shields.io/badge/Rioxarray-004D40?style=flat&logo=conda-forge&logoColor=white)](https://corteva.github.io/rioxarray/stable/)
- **Web Framework**:
  [![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

## Usage

1.  **Data Preparation**
    - Place your climate data in the `Data/` directory
    - Ensure data includes rainfall, temperature, and soil moisture measurements

2.  **Risk Assessment**

    ```python
    from risk_assesment import generate_climate_risk_assessment

    # Generate risk assessment
    results = generate_climate_risk_assessment(
        data_path='path/to/merged_data.csv',
        output_dir='risk_assessment_results',
        zone_data_path='path/to/zone_data.csv'
    )
    ```

3.  **View Results**
    - Check the `risk_assessment_results/` directory for generated reports
    - Review visualizations in the `Visualisation/` directory

## Risk Assessment Methodology

The system employs a multi-layered approach combining statistical and machine learning methods for robust climate risk assessment:

### Data Analysis & Preprocessing

- Historical climate data aggregation and baseline calculation
- Extraction of relevant climate features (rainfall, temperature, soil moisture)
- Temporal and spatial trend identification and anomaly detection

### Risk Calculation & Feature Engineering

- Calculation of indices such as Drought Index (rainfall z-score and soil moisture ratio)
- Heat stress quantification using temperature anomalies and threshold exceedance
- Flood risk assessment based on rainfall intensity and soil saturation levels

### Risk Classification & Scoring

- Classification into severity levels: Low, Mild, Moderate, Severe
- Use of Random Forest classifiers trained on engineered features for drought, heat, flood, and rainfall risks
- Weighted scoring system integrating multiple risk factors with zone-specific thresholds for localized risk evaluation

## Contributing

1.  Fork the repository
2.  Create a feature branch
3.  Commit your changes
4.  Push to the branch
5.  Create a Pull Request

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

