import folium

def plot_earthquakes(df):
    # Center map on Japan
    m = folium.Map(location=[36.0, 138.0], zoom_start=5)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            popup=f"Magnitude: {row['mag']}",
            color='red',
            fill=True
        ).add_to(m)

    m.save("outputs/earthquake_map.html")
    print("Map saved to outputs/earthquake_map.html")