import os
import config
# import subprocess

def start_job(s_lat, s_long, t_lat, t_long, exp_name):
    print(s_lat, s_long, t_lat, t_long, exp_name)
    os.system(
        (f'cd {config.PROJECT_ROOT_PATH} && '
         f'python scripts/scan.py -s1 {s_lat} -s2 {s_long} -d1 {t_lat} -d2 {t_long} '
         f'-b 64 -n {exp_name.replace(" ", "_")}')
    )