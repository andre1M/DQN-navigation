from agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DoubleDuelingDQNAgent, PERDoubleDuelingDQNAgent

from unityagents import UnityEnvironment
import torch


# pick a solver
user_input = int(
    input("Enter a number of corresponding solver to visualize (must have an appropriate saved checkpoint):\n"
          "\t1 -- Vanilla DQN\n"
          "\t2 -- Double DQN\n"
          "\t3 -- Dueling DQN\n"
          "\t4 -- Double Dueling DQN\n"
          "\t5 -- Double Dueling DQN with PER\n")
)

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

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

# initialize agent and load the weights from file
if solver_to_use[0]:
    agent = DQNAgent(state_size, action_size, seed=0)
    agent.network_local.load_state_dict(torch.load('checkpoints/checkpoint_vanilla.pth'))
elif solver_to_use[1]:
    agent = DoubleDQNAgent(state_size, action_size, seed=0)
    agent.network_local.load_state_dict(torch.load('checkpoints/checkpoint_double.pth'))
elif solver_to_use[2]:
    agent = DuelingDQNAgent(state_size, action_size, seed=0)
    agent.network_local.load_state_dict(torch.load('checkpoints/checkpoint_dueling.pth'))
elif solver_to_use[3]:
    agent = DoubleDuelingDQNAgent(state_size, action_size, seed=0)
    agent.network_local.load_state_dict(torch.load('checkpoints/checkpoint_double_dueling.pth'))
elif solver_to_use[4]:
    agent = PERDoubleDuelingDQNAgent(state_size, action_size, seed=0)
    agent.network_local.load_state_dict(torch.load('checkpoints/checkpoint_double_dueling_per.pth'))
else:
    raise RuntimeError


for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for j in range(300):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        if done:
            break

# close environment
env.close()
