from preprocessing.clean_data import load_and_clean
from preprocessing.create_grids import assign_grids
from preprocessing.make_timeseries import create_timeseries
from utils.helpers import create_sequences
from utils.prediction_map import plot_prediction_map

from models.cnn_lstm import build_model as cnn_lstm
from models.cnn_gru import build_model as cnn_gru
from models.lenet_lstm import build_model as lenet_lstm
from models.resnet_lstm import build_model as resnet_lstm

from training.train import train_and_select_best
from utils.visualize import plot_3d_surface, plot_heatmap_forecast
from utils.map_viz import plot_earthquakes

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import os


# ---------- Heatmap (3x3) ----------
def show_heatmap(pred):
    grid = pred.reshape(3, 3)

    plt.imshow(grid)
    plt.colorbar()
    plt.title("Next Day Seismic Activity (3x3)")

    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{grid[i][j]:.2f}",
                     ha='center', va='center', color='white')

    plt.show()


# ---------- 7-day Forecast ----------
def predict_future(model, last_seq, days, scalers):
    current_seq = last_seq.copy()
    predictions = []

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, 7, 9), verbose=0)

        #inverse per-grid
        pred_real = np.zeros_like(pred)
        for i in range(pred.shape[1]):
            pred_real[:, i] = scalers[i].inverse_transform(
                pred[:, i].reshape(-1, 1)
            ).flatten()

        pred_real = np.maximum(pred_real, 0)
        predictions.append(pred_real[0])

        #scale again per-grid
        pred_scaled = np.zeros_like(pred_real)
        for i in range(pred_real.shape[1]):
            pred_scaled[:, i] = scalers[i].transform(
                pred_real[:, i].reshape(-1, 1)
            ).flatten()

        current_seq = np.vstack([current_seq[1:], pred_scaled])

    return np.array(predictions)


# ---------- Classification ----------
def classify(x):
    if x < 0.2:
        return "Low"
    elif x < 0.8:
        return "Moderate"
    else:
        return "High"


# ---------- Main ----------
def main():

    print("Starting pipeline...\n")

    # Load + preprocess
    df = load_and_clean("data/japan_earthquakes.csv")
    df = assign_grids(df)
    ts = create_timeseries(df).values

    print("Time-series shape:", ts.shape)

    # FIX: Per-grid normalization
    scalers = []
    ts_scaled = np.zeros_like(ts)

    for i in range(ts.shape[1]):
        scaler = StandardScaler()
        ts_scaled[:, i] = scaler.fit_transform(ts[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)

    # Sequences
    X, y = create_sequences(ts_scaled, window=7)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Models
    model_builders = {
        "CNN_LSTM": cnn_lstm,
        "CNN_GRU": cnn_gru,
        "LeNet_LSTM": lenet_lstm,
        "ResNet_LSTM": resnet_lstm
    }

    # Load or train
    keras_path = "outputs/best_model.keras"
    h5_path = "outputs/best_model.h5"

    if os.path.exists(keras_path):
        print("Loading .keras model...")
        model = load_model(keras_path)

    elif os.path.exists(h5_path):
        print("Loading .h5 model...")
        model = load_model(h5_path, compile=False)

    else:
        print("Training model...")
        model, results = train_and_select_best(X, y, model_builders)

        print("\nModel comparison:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

    # -------- Next day prediction --------
    last_seq = X[-1]
    pred = model.predict(last_seq.reshape(1, 7, 9), verbose=0)

    # inverse per-grid
    pred_real = np.zeros_like(pred)
    for i in range(pred.shape[1]):
        pred_real[:, i] = scalers[i].inverse_transform(
            pred[:, i].reshape(-1, 1)
        ).flatten()

    pred_real = np.maximum(pred_real, 0)

    print("\nNext day prediction:")
    print(pred_real[0])

    # Classification
    labels = [classify(v) for v in pred_real[0]]
    print("Activity levels:", labels)

    # Heatmap + Map
    show_heatmap(pred_real[0])
    plot_prediction_map(pred_real[0])

    # -------- 7-day forecast --------
    future_7 = predict_future(model, last_seq, 7, scalers)

    print("\n7-day forecast:")
    print(future_7)

    # Map + visuals
    plot_earthquakes(df)

    print("\n7-day visualizations...")
    plot_3d_surface(future_7, "7-Day Forecast (3D)")
    plot_heatmap_forecast(future_7, "7-Day Forecast Heatmap")

    # Trend (center grid)
    plt.plot(future_7[:, 4])
    plt.title("7-Day Trend (Center Grid)")
    plt.xlabel("Days")
    plt.ylabel("Activity")
    plt.show()


if __name__ == "__main__":
    main()