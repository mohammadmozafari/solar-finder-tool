import os
import redis
import config
# import subprocess

def start_job(s_lat, s_long, t_lat, t_long, exp_name):
    os.system(
        (f'cd {config.PROJECT_ROOT_PATH} && '
         f'python scripts/scan.py -s1 {s_lat} -s2 {s_long} -d1 {t_lat} -d2 {t_long} '
         f'-b 64 -n {exp_name.replace(" ", "_")}')
    )
    
def add_job_to_dataset(s_lat, s_long, t_lat, t_long, exp_name):
    r = redis.StrictRedis(host='127.0.0.1', port=6379, charset="utf-8", decode_responses=True)
    if not r.ping():
        return False
    job_ids = [int(str(x).split(':')[-1]) for x in r.scan_iter('job:*')]
    new_job_id = max(job_ids) + 1
    r.hset(f'job:{new_job_id}', 'exp_name', exp_name)
    r.hset(f'job:{new_job_id}', 's_lat', s_lat)
    r.hset(f'job:{new_job_id}', 's_long', s_long)
    r.hset(f'job:{new_job_id}', 't_lat', t_lat)
    r.hset(f'job:{new_job_id}', 't_long', t_long)
    return True