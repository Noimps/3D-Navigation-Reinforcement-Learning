# 3D-Navigation-Reinforcement-Learning
This project provides scientists with platform to create a 3D environment from integer blocks in a text file, and then allows them to train RL models within this environment.
This code is open for everyone to develop with. If used for any scientific research, credit would nice, my paper with training parameter guidelines will be uploaded shortly.


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



