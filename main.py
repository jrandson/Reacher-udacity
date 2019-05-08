from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt
import torch

from agent import Agent


def run_ddpg(env, agent, brain_name, max_episodes=1000, max_steps=10000):
    scores = []

    for episode in range(1, max_episodes + 1):
        agent.reset()
        score = 0

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        for step in range(max_steps):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                break

        scores.append(score)
        mean_score = np.mean(scores[-100:])

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, mean_score, score))

        if mean_score >= 30.0:
            print("\t Model reached the score goal in {} episodes!".format(episode))
            break

    torch.save(agent.online_actor.state_dict(), "actor_model.path")
    torch.save(agent.online_critic.state_dict(), "critic_model.path")

    return scores


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64", no_graphics=True)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size)
    scores = run_ddpg(env, agent, brain_name, max_episodes=1000, max_steps=1000)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_apisodes.png")
    plt.show()

    w = 10
    mean_score = [np.mean(scores[i - w:i]) for i in range(w, len(scores))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(mean_score) + 1), mean_score)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_apisodes_smorthed.png")
    plt.show()