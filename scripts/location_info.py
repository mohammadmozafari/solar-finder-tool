# Correct order: latitude, longitude (from equator, from Greenwich)

import sys
sys.path.append('./')

import requests
import config
import glob
import pandas
import time
import os
import json

def get_location_info(latitude1, longitude1, latitude2=None, longitude2=None, key='RB8p6BHR17aAjuWIrG4soPGMxuuhmaKO', radius=200):

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
        url = f"https://api.os.uk/search/places/v1/nearest?radius={radius}&point={x1},{y1}&srs=WGS84&key={key}"

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
            org_addresses['CLASSIFICATION_CODE'] = dpa_info.get("CLASSIFICATION_CODE")
            org_addresses['CLASSIFICATION_CODE_DESCRIPTION'] = dpa_info.get("CLASSIFICATION_CODE_DESCRIPTION")
    
    if org_addresses == {}: org_addresses['Result']=f"No Organization found within {radius} meters of the given point or inside the region"

    return org_addresses

def pos_address_lookup(threshold=2):
    output = {}
    for folder in glob.glob(config.DATA_ROOT_PATH+"/*"):
        print(folder)
        with open(folder+'/'+'addr_info.json', 'r') as fp:
                addr_info = json.load(fp)
        for f in glob.glob(folder+'/confirmed_positive_images/*'):
            print(f)
            coords = f.split('/')[-1][:-4].split('_')[1:3] # extracting the location (lat, lon) from the image name
            if '(' in f: conf = f.split('/')[-1][:-4].split('_')[3][1:-1] # extracting the conf in () at the end of img name
            else: conf = 10
            latitude, longitude = coords # Correct order: latitude, longitude (from equator, from Greenwich)
            latitude, longitude = float(latitude)-0.000171, float(longitude)+0.000268 # center of the image location
            latitude, longitude = round(latitude, 6), round(longitude, 6)
            if f"{coords[0]},{coords[1]}" not in addr_info and float(conf) > threshold:
                result = get_location_info(latitude, longitude)
            else: # already looked up
                result = addr_info[f"{coords[0]},{coords[1]}"]
            for key, value in result.items():
                if key=='Result': output['('+str(latitude)+', '+str(longitude)+')'] = value
                else:
                    if 'CLASSIFICATION_CODE' in result:
                        output[key+' ['+result['CLASSIFICATION_CODE']+'] '] = value + ' ('+str(latitude)+', '+str(longitude)+')'
                        new_with_classification = f[:-4]+'_'+result['CLASSIFICATION_CODE']+f[-4:]
                        os.rename(f, new_with_classification)
                    else:
                        output[key] = value + ' ('+str(latitude)+', '+str(longitude)+')'
                break
            if f"{coords[0]},{coords[1]}" not in addr_info:
                addr_info[f"{coords[0]},{coords[1]}"] = result

        with open(folder+'/'+'addr_info.json', 'w') as fp:
            json.dump(addr_info, fp)

    return output

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Get location information using OS API.")
    # parser.add_argument("x1", type=float, help="X-coordinate")
    # parser.add_argument("y1", type=float, help="Y-coordinate")
    # parser.add_argument("--x2", type=float, help="Second X-coordinate for bounding box")
    # parser.add_argument("--y2", type=float, help="Second Y-coordinate for bounding box")
    # parser.add_argument("--key", type=str, default="RB8p6BHR17aAjuWIrG4soPGMxuuhmaKO", help="API key")

    # args = parser.parse_args()
    x = pandas.read_csv("scripts/large_roof.csv")   

    output = []
    for index, row in x[['Unnamed: 0', 'latitude', 'longitude']].iterrows():
        # print(row[0], row[1])
        time.sleep(0.1)
        result = get_location_info(row[1], row[2], None, None, 'RB8p6BHR17aAjuWIrG4soPGMxuuhmaKO')
        output += [str(int(row[0])) + ":" + str(result)]
        if index > 30: break
    with open('output.txt', 'w') as f:
        for i in output:
            f.write(str(i)+'\n')
    # result = get_location_info(args.x1, args.y1, args.x2, args.y2, args.key)
    print(result)