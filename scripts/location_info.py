# Correct order: latitude, longitude (from equator, from Greenwich)

import sys
sys.path.append('./')

import argparse
import requests
import config
import glob
import pickle

def get_location_info(latitude1, longitude1, latitude2=None, longitude2=None, key='T3nFmNQZon2G36cSPMECTTmK2GlfZkpM'):

    # sorting the inputs so regardless of choice of region it works
    if latitude2 is not None and longitude2 is not None:
        x1 = min(latitude1, latitude2)
        y1 = min(longitude1, longitude2)
        x2 = max(latitude1, latitude2)
        y2 = max(longitude1, longitude2)
    else:
        x1 = latitude1
        y1 = longitude1

    if latitude2 is not None and longitude2 is not None:
        url = f"https://api.os.uk/search/places/v1/bbox?bbox={x1},{y1},{x2},{y2}&srs=WGS84&key={key}"
    else:
        url = f"https://api.os.uk/search/places/v1/nearest?radius=1000&point={x1},{y1}&srs=WGS84&key={key}"

    response = requests.get(url)
    data = response.json()

    
    if "fault" in data:
        print(data)
        return data

    if "error" in data:
        print(f"Error: {data['error']['statuscode']} - {data['error']['message']}")
        return data

    results = data.get("results", [])
    org_addresses = {}

    for result in results:
        dpa_info = result.get("DPA", {})
        org_name = dpa_info.get("ORGANISATION_NAME")
        address = dpa_info.get("ADDRESS")

        if org_name and address:
            org_addresses[org_name] = address
    
    if org_addresses == {}: org_addresses['Result']="No Organization found within 1000 meters of the given point or inside the region"


    return org_addresses

def pos_address_lookup():
    output = {}
    for f in glob.glob(config.DATA_ROOT_PATH+"/*/confirmed_positive_images/*"):
            coords = f.split('/')[-1][:-4].split('_')[1:] # extracting the location (lat, lon) from the image name
            latitude, longitude = coords # Correct order: latitude, longitude (from equator, from Greenwich)
            latitude, longitude = float(latitude)-0.000171, float(longitude)+0.000268 # center of the image location
            latitude, longitude = round(latitude, 6), round(longitude, 6)
            result = get_location_info(latitude, longitude)
            for key, value in result.items():
                if key=='Result': output['('+str(latitude)+', '+str(longitude)+')'] = value
                else: output[key] = value + ' ('+str(latitude)+', '+str(longitude)+')'
            print("!")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get location information using OS API.")
    parser.add_argument("x1", type=float, help="X-coordinate")
    parser.add_argument("y1", type=float, help="Y-coordinate")
    parser.add_argument("--x2", type=float, help="Second X-coordinate for bounding box")
    parser.add_argument("--y2", type=float, help="Second Y-coordinate for bounding box")
    parser.add_argument("--key", type=str, default="T3nFmNQZon2G36cSPMECTTmK2GlfZkpM", help="API key")

    args = parser.parse_args()
    result = get_location_info(args.x1, args.y1, args.x2, args.y2, args.key)

    print(result)