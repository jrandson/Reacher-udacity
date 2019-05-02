import time

from unityagents import UnityEnvironment
from parallelEnv import parallelEnv
import numpy as np
import torch.optim as optim

from collections import deque
import matplotlib.pyplot as plt
import torch

from agent import Agent
from models import Actor, Critic


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


def run_ddpg(n_episodes=1000, max_t=10000, print_every=100):
    """DDQN Algorithm.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): frequency of printing information throughout iteration """

    gamma = 0.99
    scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]  # get the current state
        score = 0

        for t in range(max_t):
            print("\r {} from {} episode: {}".format(t, max_t, i_episode), end="", flush=True)
            action = agent.act(state)  # select an action

            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            agent.learn(state, action, reward, next_state, done, gamma)  # take step with agent (including learning)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step

            if done:  # exit loop if episode finished
                break

        scores.append(score)  # save most recent score

        print('\rEpisode {} \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)), end="", flush=True)

        if i_episode % print_every == 0 and len(scores) >= 100:
            print('\r Episode {}\tAverage Score (last 100 episodes) : {:.2f}'.format(i_episode, np.mean(scores[:-100])))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')

        if np.mean(scores[:-100]) >= 30.0:
            print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores[:-100])))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            break

    return scores



if __name__ == "__main__":



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

    scores = run_ddpg(n_episodes=1000, max_t=500)

    w = 10
    smorth_scores = [np.mean(scores[i-w:i]) for i in range(w, len(scores))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(smorth_scores)), smorth_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('episode_x_score.png')
    plt.show()