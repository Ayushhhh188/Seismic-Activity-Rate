import numpy as np

def create_timeseries(df):
    """
    Creates time-series data (date x grid) with improved seismic activity signal.
    """

    # Step 1: Convert time to date
    df['date'] = df['time'].dt.date

    # Step 2: Remove very small earthquakes (noise)
    df = df[df['mag'] >= 3.5]

    # Step 3: Handle missing magnitudes
    df['mag'] = df['mag'].fillna(0)

    # Step 4: Create weighted activity 
    # log1p reduces extreme values but keeps variation
    df['activity'] = np.log1p(df['mag'])

    # Step 5: Aggregate per day per grid 
    ts = df.groupby(['date', 'grid_id'])['activity'].sum().unstack(fill_value=0)

    # Step 6: Ensure all 9 grids exist
    for i in range(9):
        if i not in ts.columns:
            ts[i] = 0

    # Step 7: Sort grid columns properly
    ts = ts.sort_index(axis=1)

    # Step 8: Add small epsilon (avoid all-zero issues)
    ts += 1e-6

    return ts