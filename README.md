# Seismic Activity Prediction using Deep Learning

## Overview

This project is a deep learning–based seismic activity prediction system designed to analyze historical earthquake data and forecast regional seismic activity trends across Japan. The system combines spatial and temporal modeling using hybrid neural network architectures such as CNN-LSTM, CNN-GRU, LeNet-LSTM, and ResNet-LSTM.

The project transforms raw earthquake records into structured spatial-temporal grid representations and predicts future seismic activity using past seismic patterns. Multiple visualization techniques including heatmaps, 3D surface plots, and interactive geographic maps are used to improve interpretability of predictions.

![image alt](https://github.com/Ayushhhh188/Seismic-Activity-Rate/blob/main/outputs/Screenshot%202026-04-03%20004556.png)

*Figure 1: Pattern for Earthquakes in japan between 2001-2018*

![image alt](https://github.com/Ayushhhh188/Seismic-Activity-Rate/blob/main/outputs/Screenshot%202026-04-06%20163434.png)

*Figure 2: Predicted Seismic Activity for next 7 Days*

## Features

- Spatial-temporal seismic activity prediction
- 3×3 grid-based representation of Japan
- Hybrid deep learning architectures:
  - CNN-LSTM
  - CNN-GRU
  - LeNet-LSTM
  - ResNet-LSTM
- Automatic best-model selection using MAE
- 7-day future seismic forecasting
- Heatmap visualization
- 3D seismic activity surface plots
- Interactive Japan prediction maps using Folium
- Historical earthquake mapping
- Time-series preprocessing and normalization
- Modular project structure

## Problem Statement

Traditional earthquake prediction is extremely difficult due to the complex and dynamic nature of tectonic activity. However, identifying regional seismic activity trends can assist in seismic monitoring and risk analysis.

This project aims to:

- Analyze historical earthquake activity
- Learn spatial and temporal seismic patterns
- Predict future regional seismic activity levels
- Visualize activity trends geographically

## Dataset

The dataset used in this project is based on the:

**USGS Earthquake Catalog**  
[https://earthquake.usgs.gov/](https://earthquake.usgs.gov/)

The dataset contains:

- Earthquake timestamps
- Magnitude values
- Latitude and longitude coordinates
- Depth information

## Data Preprocessing

The preprocessing pipeline includes:

### 1. Data Cleaning
- Removal of missing or invalid values
- Datetime conversion

### 2. Magnitude Filtering
Very small earthquakes are removed to reduce noise:

```python
df = df[df['mag'] >= 3.5]
```
### 3. Spatial Grid Creation

Japan is divided into a 3×3 spatial grid.

Each earthquake event is assigned to a grid region:
text

G0 G1 G2
G3 G4 G5
G6 G7 G8

### 4. Log Transformation

Magnitude values are transformed using:
python

np.log1p(magnitude)

This:

    Reduces skewness

    Compresses extreme values

    Preserves variation

### 5. Time-Series Generation

Daily seismic activity is aggregated for each grid region.

Final structure:
Date	G0	G1	G2	G3	G4	G5	G6	G7	G8
...									
### 6. Standard Scaling

Each grid is normalized independently using StandardScaler.

Benefits:

    Stable training

    Balanced learning across regions

    Prevents dominant grids from overpowering others

Deep Learning Architecture

The project uses hybrid spatial-temporal architectures.
CNN Role

CNN layers are responsible for:

    Capturing spatial relationships between neighboring grids

    Detecting regional activity patterns

    Learning local seismic structures

Since the data is represented as 3×3 spatial grids, CNN treats each day as a small image-like representation.
LSTM / GRU Role

LSTM and GRU layers:

    Learn temporal dependencies

    Capture sequential seismic trends

    Model how activity evolves over time

The system uses the previous 7 days to predict future activity.
Models Implemented
1. CNN-LSTM
Combines CNN spatial learning with LSTM temporal learning.

2. CNN-GRU
Uses GRU instead of LSTM for faster sequential modeling.

3. LeNet-LSTM
A lightweight CNN architecture combined with LSTM.

4. ResNet-LSTM
Uses residual CNN blocks before temporal modeling.
Model Evaluation

Models are evaluated using:

Mean Absolute Error (MAE)
MAE = average absolute difference between predicted and actual values

Lower MAE indicates better performance.
Results
Model	MAE
CNN-LSTM	0.2223
CNN-GRU	0.2258
LeNet-LSTM	0.2196
ResNet-LSTM	0.2361

### LeNet-LSTM achieved the best performance due to its balanced complexity and suitability 
for small spatial grid inputs.

### Forecasting

The model performs:

    Next-day prediction

    Recursive 7-day forecasting

### Predictions represent:

    Relative seismic activity intensity

    Not exact earthquake occurrence

### Visualizations

The project includes multiple visualization methods:

    Heatmaps – Shows activity intensity across spatial grids.

    3D Surface Plots – Visualizes seismic activity trends across time and regions.

    Line Graphs – Displays activity progression over forecast periods.

    Interactive Japan Maps – Built using Folium to display:

        Historical earthquakes

        Predicted regional activity

        

### Installation
1. Clone Repository
bash: git clone https://github.com/YOUR_USERNAME/Seismic-Activity-Rate.git
      cd Seismic-Activity-Rate

2. Install Dependencies
bash: pip install -r requirements.txt

3. Run the Project
bash: python main.py

### Technologies Used

    Python

    TensorFlow / Keras

    NumPy

    Pandas

    Scikit-learn

    Matplotlib

    Folium

### Research Inspiration

This project is inspired by recent research on hybrid deep learning architectures for seismic analysis, particularly CNN-LSTM–based approaches for spatio-temporal seismic modeling.
Future Improvements

    Real-time seismic streaming

    Attention mechanisms

    Transformer-based architectures

    Larger spatial grid resolutions

    Classification-based earthquake event prediction

    Deployment as a web application

    Real-time dashboard visualization

### Conclusion

This project demonstrates the effectiveness of deep learning for modeling seismic activity trends using spatial-temporal earthquake data. By combining CNN-based spatial feature extraction with recurrent temporal modeling, the system successfully captures regional seismic behavior and generates meaningful activity forecasts.

The project also highlights the importance of preprocessing, normalization, and appropriate architecture selection for real-world seismic prediction tasks.
References

    USGS Earthquake Catalog – https://earthquake.usgs.gov/

    TensorFlow Documentation – https://www.tensorflow.org/

    Keras Documentation – https://keras.io/

    A Hybrid CNN-LSTM Architecture for Seismic Event Detection Using High-Rate GNSS Velocity Time Series – DOI: 10.3390/s26020519

    Earthquake Prediction Optimization using Deep Learning Hybrid RNN-LSTM Model for Seismicity Analysis – DOI: 10.1016/j.soildyn.2025.109432
