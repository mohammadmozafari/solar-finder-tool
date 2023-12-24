import numpy as np

def measure_meters(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    R = 6378.137; # Radius of earth in KM
    dLat = lat2 * np.math.pi / 180 - lat1 * np.math.pi / 180
    dLon = lon2 * np.math.pi / 180 - lon1 * np.math.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.math.pi / 180) * np.cos(lat2 * np.math.pi / 180) *np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000; # meters