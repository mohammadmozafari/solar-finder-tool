import sys
sys.path.append('./')

import os
import json
import redis
import shutil
import pickle
import config
import threading
import numpy as np
from pathlib import Path
from utils.utilities import measure_meters
from utils.jobs import start_job, add_job_to_dataset, check_job_disk_usage
from flask import Flask, request, render_template, abort, send_file, jsonify

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
        return jsonify({'message': 'Area diamiter more than 10K, please input a smaller area'}), 400
    success, job_id = add_job_to_dataset(s_lat, s_long, t_lat, t_long, exp_name)
    if not success:
        return jsonify({'message': 'Couln\'t submit job to redis.'}), 500
    threading.Thread(target=start_job, args=(s_lat, s_long, t_lat, t_long, exp_name, job_id)).start()
    return jsonify({'message': 'Job submitted successfully, please wait a few minutes before checking the results'}), 200

@app.route('/status')
def status():
    r = redis.StrictRedis(host='127.0.0.1', port=6379, charset="utf-8", decode_responses=True)
    jobs = [x for x in r.scan_iter('job:*')]
    jobs = sorted(jobs, reverse=True)
    jobs_stats = {}
    for j in jobs:
        jobs_stats[j] = {
            'exp_name': r.hget(j, 'exp_name'),
            'status': r.hget(j, 'status'),
            'progress': r.hget(j, 'progress'),
            'disk_usage': check_job_disk_usage(r.hget(j, 'storage_path'))
        }
    r.close()
    return jsonify(jobs_stats), 200

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
