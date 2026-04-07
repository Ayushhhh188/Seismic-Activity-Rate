import numpy as np

def assign_grids(df):
    lat_bins = np.linspace(df.latitude.min(), df.latitude.max(), 4)
    lon_bins = np.linspace(df.longitude.min(), df.longitude.max(), 4)

    df['lat_bin'] = np.digitize(df['latitude'], lat_bins) - 1
    df['lon_bin'] = np.digitize(df['longitude'], lon_bins) - 1

    df['lat_bin'] = df['lat_bin'].clip(0, 2)
    df['lon_bin'] = df['lon_bin'].clip(0, 2)

    # 3×3 grid → 9 regions
    df['grid_id'] = df['lat_bin'] * 3 + df['lon_bin']

    return df