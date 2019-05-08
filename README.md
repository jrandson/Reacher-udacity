#### Deep Deterministic Polyce Gradient 

###### Reacher

![An example of the environment to be solved](reacher.gif)
#### The Environment

In this project, an agent is trained to navigate (and collect bananas!) in a large, square world.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided 
for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to 
maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque 
applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 
In this project, the environment was solved at episode 158, when the average of the last 100 episodes was, for the first time,
igual or greater than +30. It's important to notice that the model achieves a score of +30, for the first time, at the episode
63 (Score: 30.19).


### Basic setup

For this project, You can download the from the link below. You need only select the environment that matches your operating system:

* [Linux environment: click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* [Mac OSX environemnt: click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)

Then, place the file in the Reacher-udacity/ folder and unzip (or decompress) the file. Open the main.py file and adjust the
path of the Reacher environment to the correct one: 

`env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64", no_graphics=True)`

You are encouraged to use a virtual env. You can set it using anaconda:
`$ conda create drlnd-env` and, then, activate then: `$ conda activate drlnd-env`. Finally, you can install the requirements to
this project by running `$ pip install -r requirements.txt`

After the instaltion is finished, run the main.py file by `$ python main.py` and observe the train model 
running

![An example of the environment to be solved](score_x_episodes.png)
