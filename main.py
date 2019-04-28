from unityagents import UnityEnvironment
import numpy as np
from matplotlib import pyplot as plt

from utils import device
from Policy import Policy


def print_env_info(env):
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])





if __name__ == "__main__":

    print("using device: ", device)

    env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    #print(env_info)
    print_env_info(env)
    env.close()


    # run your own policy!
    policy = Policy().to(device)
    # policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    import torch.optim as optim

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)


    # agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    #
    # scores = run_dql(env, agent)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.savefig('episodexscore.png')
    # plt.show()