import sys
sys.path.append('./')

import os
import cv2
import math
import json
import torch
import redis
import config
import argparse
import threading
import numpy as np
import pandas as pd
import glob as glob
from pathlib import Path
from leafmap import leafmap
from datetime import datetime
import torch.nn.functional as F
from config import MODEL_HEAD_PATH
from models.models import DinoBackbone
from models.models import DinoClassifier
from datautils.dataset import PipelineDataset
from torch.utils.data import Dataset, DataLoader
from utils.normalization import normalize, denormalize
from models.classifier_heads import TransformerEncoderLinearHead

if __name__ == "__main__":
    
    # Correct order: latitude, longitude (from equator, from Greenwich)
    # example: 43.464970, -80.547642, 43.475934, -80.539426          
    # example: 51.55208120092847, 0.11804166939715281, 51.554027915859244, 0.1209735951757776           (In London)
     
    # python scan.py -n debug -s1 52.45079903195844 -s2 -1.8550736679488515 -d1 52.45240797856021 -d2 -1.8517963291255226 --no-deploy

    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--source_lat", help="source location latitude", type=float)
    parser.add_argument("-s2", "--source_long", help="source location logitude", type=float)
    parser.add_argument("-d1", "--dest_lat", help="destination location latitude", type=float)
    parser.add_argument("-d2", "--dest_long", help="destination location latitude", type=float)
    parser.add_argument("-t", "--threads", help="number of threads", type=int, default=1)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=1)
    parser.add_argument("-n", "--name", help="name of the experiment", type=str, default="Test")
    parser.add_argument("-j", "--job_id", help="id of the job", type=str, default=-1)
    parser.add_argument('--dino_size', default='giant', type=str)
    parser.add_argument('--deploy', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(f'Debug Mode: {not args.deploy}')
    
    exp_path = Path(config.DATA_ROOT_PATH) / f'{args.job_id}_{args.name}'
    if not Path(exp_path).exists(): Path(exp_path).mkdir()
    image_path = exp_path / f'{args.name}.tif'
    json_path = exp_path / 'predictions.json'
    addr_info_path = exp_path / 'addr_info.json'

    unconfirmed_positive_images_dir = exp_path / 'unconfirmed_positive_images'
    if not Path(unconfirmed_positive_images_dir).exists(): Path(unconfirmed_positive_images_dir).mkdir()
    stdout_path = exp_path / 'stdout.txt'
    stderr_path = exp_path / 'stderr.txt'
    if args.deploy:
        sys.stdout = open(stdout_path, 'w')
        sys.stderr = open(stderr_path, 'w')
    
    if args.job_id != -1:
        r = redis.StrictRedis(host='127.0.0.1', port=6379, charset="utf-8", decode_responses=True)

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

    if args.job_id != -1:
        r.hset(f'job:{args.job_id}', 'pid', f'{os.getpid()}')
        r.hset(f'job:{args.job_id}', 'status', 'downloading')

    ds = PipelineDataset(source=area_source, target=area_dest, img_file_name=str(image_path), 
                         MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)
    
    backbone = DinoBackbone(dino_size=args.dino_size)
    backbone = backbone.to(device)
    backbone.eval()
    head = TransformerEncoderLinearHead(backbone.d_model, 2)
    head.load_state_dict(torch.load(MODEL_HEAD_PATH))
    head = head.to(device)

    if args.job_id != -1:
        r.hset(f'job:{args.job_id}', 'status', 'processing')

    predictions = {}
    # model.eval()
    print("[+] Scanning area patches for solar panels")
    with torch.no_grad():
        for i, x in enumerate(dl):
            print(f"\r{i+1}/{dl.__len__()}", end="")
            images = x[0].to(device)
            locations = x[1].numpy()
            
            x_feats = backbone(images)
            logits = head(x_feats)
            pred = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            size_estimates = head.get_size_estimate(x_feats, vis=False)
            # size_estimates = head.get_size_estimate(x_feats, vis=True, images=images, preds=pred, start_num=i*args.batch)
            
            pos_neg = [f'{"P" if p > 0.5 else "N"}' for p in pred]
            for j in range(len(pred)):
                predictions[i*args.batch+j] = (pos_neg[j], list(locations[j]))
                if pred[j] > 0.5: # save positive image
                    original_image = denormalize(images[j].cpu().numpy(), MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
                    original_image = original_image[:, :, :].transpose((1, 2, 0))
                    cv2.imwrite(f'{unconfirmed_positive_images_dir}/{i*args.batch+j}_{round(locations[j][0], 5)}_{round(locations[j][1], 5)}_({size_estimates[j]}).png', original_image[:, :, ::-1])

            if args.job_id != -1:
                r.hset(f'job:{args.job_id}', 'progress', f'{(i+1)/len(dl):.3f}')
    
    with open(json_path, "w") as fp:
        json.dump(predictions , fp)
    
    with open(addr_info_path, 'w')as fp:
        json.dump({} , fp)
    
    print("[+] Scan completed under name:", args.name)
    
    if args.job_id != -1:
        r.hset(f'job:{args.job_id}', 'status', 'completed')
        r.hset(f'job:{args.job_id}', 'datetime_completion', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        r.close()