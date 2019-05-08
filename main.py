import time

from unityagents import UnityEnvironment
import numpy as np
import torch.optim as optim

from collections import deque
import matplotlib.pyplot as plt
import torch

from agent import Agent
from models import Actor, Critic


def run_ddpg(n_episodes=1000, env, max_t=10000, print_every=100):
    """DDQN Algorithm.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): frequency of printing information throughout iteration """

    gamma = 0.99
    scores = []
    mean_scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]  # get the current state
        score = 0

        for t in range(max_t):
            action = agent.act(state)  # select an action

            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            # agent.learn(state, action, reward, next_state, done, gamma)  # take step with agent (including learning)
            agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step

            if done:  # exit loop if episode finished
                break

        scores.append(score)  # save most recent score

        mean_scores.append(np.mean(scores[-100:]))
        print('\r Episode {}  - Average Score (100 episodes): {}'.format(i_episode, np.mean(scores[-100:])), end="", flush=True)


        if i_episode % print_every == 0:
            print('\r Episode {}\tAverage Score (last 100 episodes) : {}'.format(i_episode, np.mean(scores[:-100:])))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if np.mean(scores[:-100]) >= 30.0:
            print('\n Environment solved in {:d} episodes \tAverage Score: {}'.format(i_episode,
                                                                                         np.mean(scores[:-100:])))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            break

    return scores, mean_scores



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

    t = time.time()
    scores, mean_scores = run_ddpg(n_episodes=1000, env=env)

    t = time.time() - t
    print("\n\tTraining got {} secs to be done.\n".format(t))

    w = 10
    smorth_scores = [np.mean(scores[i-w:i]) for i in range(w, len(scores))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(smorth_scores)), smorth_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('episode_x_score.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(mean_scores)), mean_scores)
    plt.ylabel('Mean Score (100 episodes) ')
    plt.xlabel('Episode #')
    plt.savefig('mean_score_100_episode.png')
    plt.show()