from flask import Flask, request, render_template, abort, send_file
import os
import threading 

download_script_path = "/home/adelavar/pv-extractor/img_downloader.py"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def run_download_script(s_lat, s_long, t_lat, t_long, exp_name):
    os.system(f"python {download_script_path} -s1 {s_lat} -s2 {s_long} -d1 {t_lat} -d2 {t_long} -b 64 -n {exp_name}")

# python img_downloader_mthread.py -s1 43.464970 -s2 -80.547642 -d1 43.475934 -d2 -80.539426 -b 16
@app.route('/handle_request', methods=['POST'])
def handle_request():
    exp_name = request.form['exp_name']
    s_lat = request.form['s_lat']
    s_long = request.form['s_long']
    t_lat = request.form['t_lat']
    t_long = request.form['t_long']
    threading.Thread(target=run_download_script, args=(s_lat, s_long, t_lat, t_long, exp_name,)).start()
    return "your request sent successfully, please wait a few minutes before checking the results"

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