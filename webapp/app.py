import sys
sys.path.append('./')

import os
import json
import shutil
import pickle
import config
import threading
import numpy as np
from pathlib import Path
from utils.jobs import start_job
from utils.utilities import measure_meters
from flask import Flask, request, render_template, abort, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/submit_job', methods=['POST'])
def submit_job():
    input_data = request.json
    exp_name = input_data['exp_name']
    s_lat = float(input_data['s_lat'])
    s_long = float(input_data['s_lon'])
    t_lat = float(input_data['t_lat'])
    t_long = float(input_data['t_lon'])
    if measure_meters(s_lat, s_long, t_lat, t_long) > 10000:
        return "Area diamiter more than 10K, please input a smaller area"
    threading.Thread(target=start_job, args=(s_lat, s_long, t_lat, t_long, exp_name,)).start()
    return "your request sent successfully, please wait a few minutes before checking the results"

@app.route('/data/<path:req_path>')
def serve_file(req_path):
    
    abs_path = Path(config.DATA_ROOT_PATH) / req_path

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        print("FILE________________")
        return send_file(abs_path)
    
    return abort(500)

@app.route('/display_images')
def list_subfolders():
    # Path to the directory containing subfolders
    base_folder = config.DATA_ROOT_PATH

    # Get a list of subfolders in the specified directory
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]

    # Render a template to display the list of subfolders
    return render_template('subfolder_list.html', subfolders=subfolders)

@app.route('/display_images/<subfolder>')
def display_images(subfolder):
    image_folder = Path(config.DATA_ROOT_PATH) / subfolder / 'unconfirmed_positive_images'
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    return render_template('display_images.html', images=images, subfolder_name=subfolder)


@app.route('/submit_images', methods=['POST'])
def submit_images():
    selected_choices = {key: request.form[key] for key in request.form}
    subfolder_name = request.form.get('subfolder_name')
    # Process the selected images, for example, add them to a list on the server
    print("Selected Images:", subfolder_name, selected_choices)
    
    source_path = Path(config.DATA_ROOT_PATH) / subfolder_name / 'unconfirmed_positive_images'
    confirmed_positive_path = Path(config.DATA_ROOT_PATH) / subfolder_name / 'confirmed_positive_images'
    confirmed_negative_path = Path(config.DATA_ROOT_PATH) / subfolder_name / 'confirmed_negative_images'
    if not Path(confirmed_positive_path).exists(): Path(confirmed_positive_path).mkdir() 
    if not Path(confirmed_negative_path).exists(): Path(confirmed_negative_path).mkdir()
    
    for i in selected_choices:
        if selected_choices[i] == 'negative':
            shutil.move(source_path / i, confirmed_negative_path / i)
        elif selected_choices[i] == 'positive':
            shutil.move(source_path / i, confirmed_positive_path / i)

    with open(Path(config.DATA_ROOT_PATH) / subfolder_name / 'confirmed_predictions.pkl', 'wb') as fp:
        pickle.dump(selected_choices, fp)

    return "Images submitted successfully!"
