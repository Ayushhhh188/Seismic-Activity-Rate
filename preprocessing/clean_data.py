import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # keep only needed columns
    df = df[['time', 'latitude', 'longitude', 'mag', 'depth']]

    return df

if __name__ == "__main__":
    df = load_and_clean("data/japan_earthquakes.csv")  # adjust path if needed
    print(df.head())
    print(df.shape)