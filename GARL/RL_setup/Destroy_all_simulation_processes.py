import subprocess
import os
import time
import signal
import subprocess


"""
When anything goes south during developing, run this script to make sure all hidden/side/server processes get closed

If not done, and too many processes are running, you ram is going to get full and things will go south even more (linux gonna restart or shutdown)



"""
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
