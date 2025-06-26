import subprocess
import os
import time
import signal
import subprocess


def close_sim():


    # grab the whole `ps aux` output as text
    ps_output = subprocess.check_output(["ps", "aux"], text=True)

    # split into lines and filter
    matches = [
        line for line in ps_output.splitlines()
        if ("gz " in line or "sim_vehicle.py " in line) and "grep" not in line
    ]

    for line in matches:
        x = int(line.split()[1])
        print(x)
        os.killpg(os.getpgid(x), signal.SIGKILL)  # Forcefully kill the process group

    

close_sim()
