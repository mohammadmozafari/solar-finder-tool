import os
import redis
import config
import subprocess
from pathlib import Path
from datetime import datetime
# import subprocess

def start_job(s_lat, s_long, t_lat, t_long, exp_name, job_id):
    os.system(
        (f'cd {config.PROJECT_ROOT_PATH} && '
         f'python scripts/scan.py -s1 {s_lat} -s2 {s_long} -d1 {t_lat} -d2 {t_long} '
         f'-b 64 -n {exp_name.replace(" ", "_")} --job_id {job_id}')
    )
    
def add_job_to_dataset(s_lat, s_long, t_lat, t_long, exp_name):
    r = redis.StrictRedis(host='127.0.0.1', port=6379, charset="utf-8", decode_responses=True)
    if not r.ping():
        return False, None
    job_ids = [int(x.split(':')[-1]) for x in r.scan_iter('job:*')]
    new_job_id = 0 if len(job_ids) == 0 else max(job_ids) + 1
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r.hset(f'job:{new_job_id}', 'exp_name', exp_name)
    r.hset(f'job:{new_job_id}', 's_lat', s_lat)
    r.hset(f'job:{new_job_id}', 's_long', s_long)
    r.hset(f'job:{new_job_id}', 't_lat', t_lat)
    r.hset(f'job:{new_job_id}', 't_long', t_long)
    r.hset(f'job:{new_job_id}', 'storage_path', f'data/{new_job_id}_{exp_name}')
    r.hset(f'job:{new_job_id}', 'pid', 'null')
    r.hset(f'job:{new_job_id}', 'datetime_submission', dt_string)
    r.hset(f'job:{new_job_id}', 'datetime_completion', 'null')
    r.hset(f'job:{new_job_id}', 'status', 'pending')
    r.hset(f'job:{new_job_id}', 'progress', 'null')
    return True, new_job_id

def check_job_disk_usage(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    folder_name = path.split('/')[-1]
    path = Path(config.DATA_ROOT_PATH) / folder_name
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')