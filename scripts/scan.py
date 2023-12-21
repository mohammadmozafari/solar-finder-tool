import sys
sys.path.append('./')

import os
import cv2
import math
import json
import torch
import config
import argparse
import threading
import numpy as np
import pandas as pd
import glob as glob
from pathlib import Path
from leafmap import leafmap
from models.models import DinoClassifier
from datautils.dataset import PipelineDataset
from torch.utils.data import Dataset, DataLoader
from utils.normalization import normalize, denormalize

if __name__ == "__main__":
    
    # Correct order: latitude, longitude (from equator, from Greenwich)
    # example: 43.464970, -80.547642 - 43.475934, -80.539426          
    # example: 51.55208120092847, 0.11804166939715281, 51.554027915859244, 0.1209735951757776           (In London)
     
    # python img_downloader.py -s1 43.464970 -s2 -80.547642 -d1 43.475934 -d2 -80.539426 -b 16 -n img

    # images_folder = '/home/adelavar/pv-extractor/pipeline/downloaded_images/'
    # json_folder = '/home/adelavar/pv-extractor/pipeline/predictions/'
    # if not Path(images_folder).exists(): Path(images_folder).mkdir()
    # if not Path(json_folder).exists(): Path(json_folder).mkdir()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--source_lat", help="source location latitude", type=float)
    parser.add_argument("-s2", "--source_long", help="source location logitude", type=float)
    parser.add_argument("-d1", "--dest_lat", help="destination location latitude", type=float)
    parser.add_argument("-d2", "--dest_long", help="destination location latitude", type=float)
    parser.add_argument("-t", "--threads", help="number of threads", type=int, default=1)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=1)
    parser.add_argument("-n", "--name", help="name of the experiment", type=str, default="Test")
    args = parser.parse_args()
    
    exp_path = Path(config.DATA_ROOT_PATH) / args.name
    if not Path(exp_path).exists(): Path(exp_path).mkdir()
    image_path = exp_path / f'{args.name}.tif'
    json_path = exp_path / 'predictions.json'
    positive_images_dir = exp_path / 'positive_images'
    if not Path(positive_images_dir).exists(): Path(positive_images_dir).mkdir()
    stdout_path = exp_path / 'stdout.txt'
    stderr_path = exp_path / 'stderr.txt'
    sys.stdout = open(stdout_path, 'w')
    sys.stderr = open(stderr_path, 'w')

    ### TO DO ###
    # Sort locations #d

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # sorting the inputs so regardless of choice of region it works
    source_lat = min(args.source_lat, args.dest_lat)
    source_long = min(args.source_long, args.dest_long)
    dest_lat = max(args.source_lat, args.dest_lat)
    dest_long = max(args.source_long, args.dest_long)

    area_source, area_dest = (source_lat, source_long), (dest_lat, dest_long)
    length, width = np.array(area_dest) - np.array(area_source)
    num_threads = args.threads

    ds = PipelineDataset(source=area_source, target=area_dest, img_file_name=str(image_path), 
                         MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    model_head = torch.nn.Sequential(
        torch.nn.Linear(3072, 700),
        torch.nn.ReLU(),
        torch.nn.Linear(700, 2))
    model_head.load_state_dict(torch.load(config.MODEL_HEAD_PATH))
    model = DinoClassifier(mode='giant', head=model_head)
    model = model.to(device)

    predictions = {}
    # model.eval()
    print("[+] Scanning area patches for solar panels")
    with torch.no_grad():
        for i, x in enumerate(dl):
            print(f"\r{i+1}/{dl.__len__()}", end="")
            images = x[0].to(device)
            locations = x[1].numpy()
            pred = model(images)
            pred = torch.nn.functional.softmax(pred, dim=-1)[:, 1]
            pos_neg = [f'{"P" if p > 0.5 else "N"}' for p in pred]
            for j in range(len(pred)):
                predictions[i*args.batch+j] = (pos_neg[j], list(locations[j]))
                if pred[j] > 0.5: # save positive image
                    original_image = denormalize(images[j].cpu().numpy(), MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
                    original_image = original_image[:, :, :].transpose((1, 2, 0))
                    cv2.imwrite(f'{positive_images_dir}/{i*args.batch+j}_{round(locations[j][0], 5)}_{round(locations[j][1], 5)}.png', original_image[:, :, ::-1])
    
    with open(json_path, "w") as fp:
        json.dump(predictions , fp)
    
    print("[+] Scan completed under name:", args.name)
    os.remove(str(image_path))
