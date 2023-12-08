import threading
# pip install segment-geospatial
from leafmap import leafmap
import torch
import os
from pathlib import Path
import numpy as np
import json
import math
import pandas as pd
import glob as glob
import argparse
import cv2
from datautils.dataset import PipelineDataset
from torch.utils.data import Dataset, DataLoader
from models.models import DinoClassifier
from utils.normalization import normalize, denormalize

if __name__ == "__main__":
    # Correct order: latitude, longitude (from equator, from Greenwich)
    # example: 43.464970, -80.547642 - 43.475934, -80.539426           
    # python img_downloader.py -s1 43.464970 -s2 -80.547642 -d1 43.475934 -d2 -80.539426 -b 16 -n img

    images_folder = '/home/adelavar/pv-extractor/pipeline/downloaded_images/'
    json_folder = '/home/adelavar/pv-extractor/pipeline/predictions/'
    if not Path(images_folder).exists(): Path(images_folder).mkdir()
    if not Path(json_folder).exists(): Path(json_folder).mkdir()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--source_lat", help="source location latitude", type=float)
    parser.add_argument("-s2", "--source_long", help="source location logitude", type=float)
    parser.add_argument("-d1", "--dest_lat", help="destination location latitude", type=float)
    parser.add_argument("-d2", "--dest_long", help="destination location latitude", type=float)
    parser.add_argument("-t", "--threads", help="number of threads", type=int, default=1)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=1)
    parser.add_argument("-n", "--name", help="name of the experiment", type=str, default="Test")
    args = parser.parse_args()

    ### TO DO ###
    # Sort locations #d

    if not Path(images_folder+args.name).exists(): Path(images_folder+args.name).mkdir()

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

    save_path = Path('/home/adelavar/pv-extractor/pipeline/outputs')
    if not save_path.exists(): save_path.mkdir()
    ds = PipelineDataset(source=area_source, target=area_dest, img_file_name=f"/home/adelavar/pv-extractor/pipeline/{args.name}.tif", 
                         MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    model_head = torch.nn.Sequential(
        torch.nn.Linear(3072, 700),
        torch.nn.ReLU(),
        torch.nn.Linear(700, 2))
    old_ckpt_path = '/home/adelavar/pv-extractor/pipeline/checkpoints/dinov2_mlp2_160k.model'
    checkpoint = torch.load(old_ckpt_path)
    model_head.load_state_dict(checkpoint)
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
                    cv2.imwrite(f'{(images_folder+args.name)}/{i*args.batch+j}_{round(locations[j][0], 5)}_{round(locations[j][1], 5)}.png', original_image[:, :, ::-1])
    
    with open(json_folder+args.name, "w") as fp:
        json.dump(predictions , fp)
    
    print("[+] Scan completed under name:", args.name)
    
    os.remove(f"/home/adelavar/pv-extractor/pipeline/{args.name}.tif")
