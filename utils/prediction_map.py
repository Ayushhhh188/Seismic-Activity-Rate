def plot_prediction_map(pred):
    import folium
    import numpy as np

    m = folium.Map(location=[36.0, 138.0], zoom_start=5)

    # create 3×3 grid coordinates dynamically
    lat_vals = np.linspace(34, 40, 3)
    lon_vals = np.linspace(135, 142, 3)

    idx = 0
    for lat in lat_vals:
        for lon in lon_vals:
            value = float(pred[idx])

            if value < 0.2:
                color = "green"
            elif value < 0.8:
                color = "orange"
            else:
                color = "red"

            folium.Circle(
                location=[lat, lon],
                radius=30000 + value * 50000,
                popup=f"G{idx}: {value:.2f}",
                color=color,
                fill=True,
                fill_opacity=min(value / 2, 1)
            ).add_to(m)

            idx += 1

    m.save("outputs/prediction_map.html")