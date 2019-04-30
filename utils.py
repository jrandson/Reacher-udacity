# from parallelEnv import parallelEnv
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
# from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import random as rand


RIGHT=4
LEFT=5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return img

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)


# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(), \
        lambda x: patch.set_data(frames[x]), frames = len(frames), interval=30)
    
    display(display_animation(fanim, default_mode='once'))
    
# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=2000, preprocess=None, nrand=5):
    env.reset()

    # star game
    env.step(1)
    
    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT,LEFT]))
        frame2, reward2, is_done, _ = env.step(0)
    
    anim_frames = []
    
    for _ in range(time):
        
        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)
        
        # RIGHT = 4, LEFT = 5
        action = RIGHT if rand.random() < prob else LEFT
        frame1, _, is_done, _ = env.step(action)
        frame2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if is_done:
            break
    
    env.close()
    
    animate_frames(anim_frames)
    return 



# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into 
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1,fr2])
        
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
        action_list, reward_list


def collect_trajectories_unity(env, policy, n=20, tmax=200, nrand=5):
    """

    :param env:
    :param policy:
    :param n:
    :param tmax:
    :param nrand:
    :return:
    """

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]

    print("collecting trajectories")
    for _ in range(tmax):
        print("\r{}/{}".format(_, tmax), end="", flush=True)

        actions = np.random.randn(n, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        state_size = next_states.shape[1]
        next_states = np.reshape(next_states, (n, 1, state_size))
        batch_input = torch.from_numpy(next_states).float().to(device)
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        if np.any(dones):  # exit loop if episode finished
            break

    # return pi_theta, states, actions, rewards, probability
    return probs, next_states, actions, rewards


def collect_trajectories_ppo(env, policy, tmax=200, nrand=5):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    # number of parallel instances
    n = len(env_info.agents)
    print(f"num agents:{n}")

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    log_prob_list = []
    action_list = []

    states = env_info.vector_observations  # get the current state (for each agent)
    states = torch.from_numpy(states).float().to(device)

    for t in range(tmax):

        states = states.unsqueeze(0)
        action = policy.forward(states)

        log_probs = torch.tensor(np.random.rand(1,4), dtype=torch.float, device=device)

        # To use np and the environment as follows, tranfer everything into numpy
        action = action.cpu().detach().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        action = np.clip(action, -1, 1)  # all actions between -1 and 1

        env_info = env.step(action)[brain_name]  # send all actions to tne environment

        # remove [0] for multiple agents
        next_state = env_info.vector_observations  # get next state (for each agent)
        reward = env_info.rewards  # get reward (for each agent)
        done = env_info.local_done  # see if episode finished

        # store the result in form of numpy & list
        state_list.append(states)
        reward_list.append(reward)
        log_prob_list.append(log_probs)
        action_list.append(action)

        states = next_state

        # print('states middle',type(states))
        # print('states middle:',states)

        # return states from list to be tensor
        # states = torch.FloatTensor(states)
        states = torch.from_numpy(states).float().to(device)
        # print('states after',type(states))
        # print('states after:',states)

        if done:
            break

    # stop if any of the trajectories is done
    # we want all the lists to be retangular
    '''
    if is_done.any():
        break
    '''

    # return pi_theta, states, actions, rewards, probability
    return log_prob_list, state_list, \
           action_list, reward_list


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):
    """
     return sum of log-prob divided by T
     same thing as -policy_loss
    :param policy:
    :param old_probs:
    :param states:
    :param actions:
    :param rewards:
    :param discount:
    :param beta:
    :return:
    """

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

    ratio = new_probs/old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(ratio*rewards + beta*entropy)

    

def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):
    """
     clipped surrogate function
     similar as -policy_loss for REINFORCE, but for PPO
    :param policy:
    :param old_probs:
    :param states:
    :param actions:
    :param rewards:
    :param discount:
    :param epsilon:
    :param beta:
    :return:
    """

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
    
    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)

    
