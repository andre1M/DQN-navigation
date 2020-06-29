from agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DoubleDuelingDQNAgent, PERDoubleDuelingDQNAgent
from utilities import train, plot_scores

from unityagents import UnityEnvironment
import torch

import os


# pick a solver
user_input = int(input("Enter a number of corresponding solver to use:\n"
                       "\t1 -- Vanilla DQN\n"
                       "\t2 -- Double DQN\n"
                       "\t3 -- Dueling DQN\n"
                       "\t4 -- Double Dueling DQN\n"
                       "\t5 -- Double Dueling DQN with PER\n"))

allowed_choices = [1, 2, 3, 4, 5]

# ["Vanilla DQN", "DoubleDQN", "DuelingDQN", "DoubleDuelingDQN", "DoubleDuelingDQN with PER"]
solver_to_use = [0, 0, 0, 0, 0]

# check if the user input is valid
for idx, choice in enumerate(allowed_choices):
    if user_input == choice:
        solver_to_use[idx] = 1
if sum(solver_to_use) != 1:
    raise ValueError(f"Invalid user input. Allowed inputs {allowed_choices}.")


# initialize environment
env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


if solver_to_use[0]:
    # # #  --------- Vanilla DQN agent --------- # # #
    print('\nVanilla DQN')

    # initialize agent
    agent = DQNAgent(state_size, action_size, seed=0)

    # train with linear epsilon decrease
    scores = train(agent, env, n_episodes=2000, eps_start=1, eps_end=0.1, eps_decay=0.995, decline='lin')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # save network weights
    torch.save(agent.network_local.state_dict(), 'checkpoints/checkpoint_vanilla.pth')

    if not os.path.exists('plots'):
        os.mkdir('plots')

    # plot the scores
    plot_scores(scores, filename='plots/plot_vanilla.png')

    # close environment
    env.close()
    # # #  --------- Vanilla DQN agent --------- # # #


if solver_to_use[1]:
    # # #  --------- Double DQN agent --------- # # #
    print('\nDouble DQN')

    # initialize agent
    agent = DoubleDQNAgent(state_size, action_size, seed=0)

    # reset environment
    env.reset(train_mode=True)

    # train with linear epsilon decrease
    scores = train(agent, env, n_episodes=2000, eps_start=1, eps_end=0.1, eps_decay=0.995, decline='lin')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # save network weights
    torch.save(agent.network_local.state_dict(), 'checkpoints/checkpoint_double.pth')

    if not os.path.exists('plots'):
        os.mkdir('plots')

    # plot the scores
    plot_scores(scores, filename='plots/plot_double.png')
    # # #  --------- Double DQN agent --------- # # #


if solver_to_use[2]:
    # # #  --------- Dueling DQN agent --------- # # #
    print('\nDueling DQN')

    # initialize agent
    agent = DuelingDQNAgent(state_size, action_size, seed=0)

    # reset environment
    env.reset(train_mode=True)

    # train with linear epsilon decrease
    scores = train(agent, env, n_episodes=2000, eps_start=1, eps_end=0.1, eps_decay=0.995, decline='lin')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # save network weights
    torch.save(agent.network_local.state_dict(), 'checkpoints/checkpoint_dueling.pth')

    if not os.path.exists('plots'):
        os.mkdir('plots')

    # plot the scores
    plot_scores(scores, filename='plots/plot_dueling.png')

    # close environment
    env.close()
    # # #  --------- Dueling DQN agent --------- # # #


if solver_to_use[3]:
    # # #  --------- Double Dueling DQN agent --------- # # #
    print('\nDouble Dueling DQN')

    # initialize agent
    agent = DoubleDuelingDQNAgent(state_size, action_size, seed=0)

    # reset environment
    env.reset(train_mode=True)

    # train with linear epsilon decrease
    scores = train(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.1, eps_decay=0.995, decline='lin')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # save network weights
    torch.save(agent.network_local.state_dict(), 'checkpoints/checkpoint_double_dueling.pth')

    if not os.path.exists('plots'):
        os.mkdir('plots')

    # plot the scores
    plot_scores(scores, filename='plots/plot_double_dueling.png')

    # close environment
    env.close()
    # # #  --------- Double Dueling DQN agent --------- # # #


if solver_to_use[4]:
    # # #  --------- Double Dueling DQN agent with PER --------- # # #
    print('\nDouble Dueling DQN with PER')

    # initialize agent
    agent = PERDoubleDuelingDQNAgent(state_size, action_size, seed=0)

    # reset environment
    env.reset(train_mode=True)

    # train with linear epsilon decrease
    scores = train(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.1, eps_decay=0.995, decline='lin')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # save network weights
    torch.save(agent.network_local.state_dict(), 'checkpoints/checkpoint_double_dueling_per.pth')

    if not os.path.exists('plots'):
        os.mkdir('plots')

    # plot the scores
    plot_scores(scores, filename='plots/plot_double_dueling_per.png')

    # close environment
    env.close()
    # # #  --------- Double Dueling DQN agent with PER --------- # # #
