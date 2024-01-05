#!/bin/bash

export PATH=$PATH:/home/adelavar/anaconda3/bin
eval "$(conda shell.bash hook)"
conda activate /home/adelavar/anaconda3/envs/x
python -m flask --app webapp/app run --host 0.0.0.0