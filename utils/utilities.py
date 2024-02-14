import numpy as np
import json
import csv

def measure_meters(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    R = 6378.137; # Radius of earth in KM
    dLat = lat2 * np.math.pi / 180 - lat1 * np.math.pi / 180
    dLon = lon2 * np.math.pi / 180 - lon1 * np.math.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.math.pi / 180) * np.cos(lat2 * np.math.pi / 180) *np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000; # meters

def save_pos_addr_json_to_csv(json_file, csv_file='webapp/pos_lookup.csv'):

    # Write data to CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Name', 'Classification Code', 'Address', 'Location'])
        # Iterate over JSON data and write rows to CSV
        for key, value in json_file.items():
            # Parse key to extract name and classification code
            name, classification_code = key.strip().split('[')
            classification_code = classification_code.strip('] ')
            # Parse value to extract address and location
            address, location = value.rsplit('(', 1)
            location = location.strip(') ')
            # Write row to CSV
            writer.writerow([name.strip(), classification_code, address.strip(), location.strip()])