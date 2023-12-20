import os
import json
import shutil
import pickle
import threading
import numpy as np
from pathlib import Path
from webapp.status import status
from flask import Flask, request, render_template, abort, send_file

download_script_path = "/home/adelavar/pv-extractor/img_downloader.py"

app = Flask(__name__)

@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('home.html')

def run_download_script(s_lat, s_long, t_lat, t_long, exp_name):
    os.system(f"python {download_script_path} -s1 {s_lat} -s2 {s_long} -d1 {t_lat} -d2 {t_long} -b 64 -n {exp_name.replace(' ', '_')}")

# python img_downloader_mthread.py -s1 43.464970 -s2 -80.547642 -d1 43.475934 -d2 -80.539426 -b 16
@app.route('/handle_request', methods=['POST'])
def handle_request():
    
    # input_data = json.loads(request.json)
    input_data = request.json
    exp_name = input_data['exp_name']
    s_lat = float(input_data['s_lat'])
    s_long = float(input_data['s_lon'])
    t_lat = float(input_data['t_lat'])
    t_long = float(input_data['t_lon'])
    
    if measure(s_lat, s_long, t_lat, t_long) > 10000:
        return "Area diamiter more than 10K, please input a smaller area"
    
    threading.Thread(target=run_download_script, args=(s_lat, s_long, t_lat, t_long, exp_name,)).start()
    return "your request sent successfully, please wait a few minutes before checking the results"

@app.route('/status')
def get_status():
    return status.get_status()

@app.route('/downloaded_images', defaults={'req_path': ''})
@app.route('/downloaded_images/<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = '/home/adelavar/pv-extractor/pipeline/downloaded_images'

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)
    print("DIR", (BASE_DIR, req_path))

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        print("FILE________________")
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template('image_directories.html', files=files)

@app.route('/display_images')
def list_subfolders():
    # Path to the directory containing subfolders
    base_folder = '/home/adelavar/pv-extractor/pipeline/downloaded_images/'

    # Get a list of subfolders in the specified directory
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]

    # Render a template to display the list of subfolders
    return render_template('subfolder_list.html', subfolders=subfolders)

@app.route('/display_images/<subfolder>')
def display_images(subfolder):
    image_folder = '/home/adelavar/pv-extractor/pipeline/downloaded_images/'+subfolder
    images = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    return render_template('display_images.html', images=images, subfolder_name=subfolder)


@app.route('/submit_images', methods=['POST'])
def submit_images():
    selected_choices = {key: request.form[key] for key in request.form}
    subfolder_name = request.form.get('subfolder_name')
    # Process the selected images, for example, add them to a list on the server
    print("Selected Images:", subfolder_name, selected_choices)

    source_path = "/home/adelavar/pv-extractor/pipeline/downloaded_images/"+subfolder_name+"/"
    destination_path = "/home/adelavar/pv-extractor/pipeline/labeled_images/"+subfolder_name
    if not Path(destination_path).exists(): Path(destination_path).mkdir()
    if not Path(destination_path+"/negative").exists(): Path(destination_path+"/negative").mkdir()
    if not Path(destination_path+"/positive").exists(): Path(destination_path+"/positive").mkdir()
    
    for i in selected_choices:
        if selected_choices[i] == 'negative':
            shutil.move(source_path+i, destination_path+"/negative/"+i)
        elif selected_choices[i] == 'positive':
            shutil.move(source_path+i, destination_path+"/positive/"+i)

    with open('/home/adelavar/pv-extractor/pipeline/predictions/'+subfolder_name+".pos", "wb") as fp:   #Pickling
      pickle.dump(selected_choices, fp)

    # with open("pv-extractor/pipeline/predictions/"+subfolder_name+".pos", "rb") as fp:   #Unpickling
    #   b = pickle.load(fp)

    return "Images submitted successfully!"

    

def measure(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    R = 6378.137; # Radius of earth in KM
    dLat = lat2 * np.math.pi / 180 - lat1 * np.math.pi / 180
    dLon = lon2 * np.math.pi / 180 - lon1 * np.math.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.math.pi / 180) * np.cos(lat2 * np.math.pi / 180) *np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000; # meters