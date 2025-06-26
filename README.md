# 3D-Navigation-Reinforcement-Learning
This project provides scientists with simple platform to create a 3D environment from integer blocks in a text file, and then allows them to train RL models within this environment.
This code is open for everyone to develop with. If used for any scientific research, credit would nice, my paper with training parameter guidelines will be uploaded shortly.

MAIN REPOSITORY -- TRAINING IN AN IDEALISED ENVIRONMENT.
BUILDING A ROOM:

legend:

0 = free empty space

2 = Obscured space

- Build a room by creating 2D grids of 0's and 2's and state their layer above each grid. 
- The environment will layer the layers in the load_room function.
- Always state the room sizes (Width, Depth, Height) in the room script like: Size=w,d,h
- You can add Goal and Start positions to a room by stating: Start position=x,y,z and Goal=x,y,z


Create a Directory with rooms you want to train on and Set the ROOMS variable to that path in the LSTM or Train train file.

Run python -m train.lstm or train.train to start training. 

- Hyperparams can be altered in the train files.
- Reward values and environment parameters can be altered in the environment file(s).


------------------------------------------------------------------------------------------------------------------------

GARL - Ardupilot Gazebo Reinforcement learning


To be able to use this repository, one should acquire some domain knowledge on Gazebo, Ardupilot, Mavlink, GroundControl stations and drone-kit python.

The following Checklist of installs and knowledge are fundamental for using the physics engine and controlling the drone through reinforcement learning:

    - Install a dual-booth version of Ubuntu 22.04 jammy (with at least 100GB of diskspace partition pointed to Ubuntu)

    - Install Gazebo Harmonic 8.9: https://gazebosim.org/docs/harmonic/getstarted/

    - Install Ardupilot-Gazebo Plugin Repo: https://github.com/ArduPilot/ardupilot_gazebo/blob/main/README.md

    - Install dronekit: https://dronekit-python.readthedocs.io/en/latest/guide/quick_start.html#installation

    - Understanding Mavlink messaging (long messages can be encoded using Dronekit, examples are in the Sim.py code): https://ardupilot.org/dev/docs/mavlink-commands.html

    - pip install all pip dependencies and import modules

    - running: cd ./GARL/RL_setup
        -  python -m train.train_single

IMPORTANT: 

- Install both ardupilot and the ardupilot-gazebo repository in the GARL folder.

- In the Ardupilot Gazebo Repository: 

    - paste the custom made iris_with_lidar folder in the models folder.
    - Take the custom made worlds in the worlds folder and paste them allongside the default worlds in the worlds folder (Remove any you dont want to train with by placing then in another folder - escpecially those that dont contain a copter....)


    - Take the Custom parameters: gazebo-iris-gimbal.parm 
        -   place it in the config folder.


TRAINING NOTES:

- The parralel training functions as an example. Use the single environment unless you are 100% certain neither gazebo, nor Ardupilot, nor drone-kit, nor Ubuntu crash or perform bad when launching multiple instances. Further more read the instructions below and make sure your respawn method is on point before launching multiple workers.

- Currently the Computer Vision script for detecting april tags in not enabled. Enable it for goal finding tasks.

- Currently MAIN environment script: 'SingleEnv.py' contains a respawn function that calls a Teleport function in '     SingleSimulation.py' for respawning the agent without resetting the whole environment

    - Remove the respawn utility when using any mode that requires GPS or EKF, the drone will go rogue after a couple respawns with any GPS using modes. --> disable GPS in the params, alter all the modes to non gps modes and alter Do-action to use only RC input actions.


Current advice: 

 -  train in the idealised environment 
 -  Rewrite drone-kit actions to RC inputs with the same effect as the actions in the idealised environment -- Move the cell size and rotate a certain degree.

-   Test in the gazebo environment,


Or:

- Further develop the respawn Utility and finetune the parallel environments (must use RC inputs and mode = GUIDED_NOGPS)

- If respawns happen under 3 seconds, neither drone goes rogue nor does your computer crash?
    -   Congratulations, you can now train directly inside the Gazebo physics engine.











