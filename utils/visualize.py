import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# 🔥 3D Surface Plot
def plot_3d_surface(data, title="Forecast 3D Visualization"):
    """
    data shape: (days, 4 grids)
    """

    days = np.arange(data.shape[0])
    grids = np.arange(data.shape[1])

    X, Y = np.meshgrid(grids, days)
    Z = data

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel("Grid ID")
    ax.set_ylabel("Days")
    ax.set_zlabel("Seismic Activity")
    ax.set_title(title)

    fig.colorbar(surf)

    plt.show()


# 🔥 Seaborn Heatmap (clean + interpretable)
def plot_heatmap_forecast(data, title="Forecast Heatmap"):
    plt.figure(figsize=(10, 4))
    sns.heatmap(data, cmap="magma", annot=True, fmt=".2f")

    plt.xlabel("Grid")
    plt.ylabel("Day")
    plt.title(title)

    plt.show()