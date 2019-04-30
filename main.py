import time

from unityagents import UnityEnvironment
from parallelEnv import parallelEnv
import numpy as np
from matplotlib import pyplot as plt
import progressbar as pb
import torch.optim as optim

from utils import device, collect_trajectories, collect_trajectories_unity, collect_trajectories_ppo, \
    surrogate, clipped_surrogate
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


def experiment_environment(env, num_agents, action_size):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def clipped_surrigate_train():

    # keep track of how long training takes
    # WARNING: running through all 800 episodes will take 30-45 minutes

    # training loop max iterations
    num_episodes = 500

    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    epsilon = 0.1
    beta = .01
    tmax = 320
    SGD_epoch = 6

    # keep track of progress
    mean_rewards = []

    for e in range(num_episodes):

        # collect trajectories
        old_probs, states, actions, rewards = \
            collect_trajectories(envs, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # gradient ascent step
        for _ in range(SGD_epoch):
            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

            # L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
            #                                  epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    timer.finish()

def surrigate_train(envs, optimizer, policy, num_episodes):
    from parallelEnv import parallelEnv
    import numpy as np
    # WARNING: running through all 800 episodes will take 30-45 minutes

    # widget = ['training loop: ', pb.Percentage(), ' ',
    #           pb.Bar(), ' ', pb.ETA()]
    # timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()

    discount_rate = .99
    beta = .01
    tmax = 320

    # keep track of progress
    mean_rewards = []

    for e in range(num_episodes):

        # collect trajectories
        old_probs, states, actions, rewards = collect_trajectories_ppo(envs, policy, tmax=tmax)
        print(len(old_probs))
        print(len(states))
        print(len(actions))
        print(len(rewards))

        print(actions)
        print()
        print(np.shape(old_probs))
        print(old_probs)
        print()
        print(rewards)
        exit()

        total_rewards = np.sum(rewards, axis=0)

        L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("\n\r Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)), end="", flush=True)
            print(total_rewards)

        print("\r {}".format(e+1), end="", flush=True)


    return mean_rewards



if __name__ == "__main__":

    print("using device: ", device)

    env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64", no_graphics=True)

    # initialize environment
    # env = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]


    # run your own policy!
    policy = Policy().to(device)
    # policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    num_episodes = 200
    mean_rewards = surrigate_train(env, optimizer, policy, num_episodes)


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