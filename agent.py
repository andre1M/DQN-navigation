from model import DQN, DuelingDQN

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import namedtuple, deque


# # # GLOBAL VARIABLES # # #
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 32             # batch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 2.5e-4                 # learning rate
UPDATE_EVERY = 4            # how often to update the network
# # # GLOBAL VARIABLES # # #


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """
    Interacts with and learns from the environment.
    Vanilla DQN.
    """

    def __init__(self, state_size: int, action_size: int, seed: int):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.network_local = DQN(state_size, action_size, seed).to(DEVICE)
        self.network_target = DQN(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action: int, reward: float, next_state, done):
        """
        Save experiences in the replay memory and check if it's time to learn.

        :param state: (array_like) current state;
        :param action: action taken;
        :param reward: reward received;
        :param next_state: (array_like) next state;
        :param done: terminal state indicator; int or bool.
        """

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Increment time step and compare it to the network update frequency
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Check if there is enough samples in the memory to learn
            if len(self.memory) > BATCH_SIZE:
                # sample experiences from memory
                experiences = self.memory.sample()
                # learn from sampled experiences
                self.learn(experiences, GAMMA)

    def act(self, state, eps: float = 0.):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state
        :param eps: epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma: float):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        :param gamma: discount factor.
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.network_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau: float):
        """
        Soft update model parameters,
        θ_target = τ*θ_local + (1 - τ)*θ_target.

        :param local_model: (PyTorch model) weights will be copied from;
        :param target_model: (PyTorch model) weights will be copied to;
        :param tau: interpolation parameter.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DoubleDQNAgent(DQNAgent):
    """
    Interacts with and learns from the environment.
    Double DQN.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        super().__init__(state_size, action_size, seed)

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        :param gamma: discount factor.
        """

        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.network_local(states).gather(1, actions)

        # Get next actions based on local network
        next_actions = self.network_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model based on local model next actions
        Q_targets_next = self.network_target(next_states).detach().gather(1, next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)


class DuelingDQNAgent(DQNAgent):
    """
    Interacts with and learns from the environment.
    Dueling DQN.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        super().__init__(state_size, action_size, seed)

        # Q-Network
        self.network_local = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.network_target = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)


class DoubleDuelingDQNAgent(DoubleDQNAgent):
    """
    Interacts with and learns from the environment.
    Double Dueling DQN.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        super().__init__(state_size, action_size, seed)

        # Q-Network
        self.network_local = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.network_target = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)


class PERDoubleDuelingDQNAgent(DoubleDuelingDQNAgent):
    """
    Interacts with and learns from the environment.
    Double Dueling DQN with prioritized experience replay.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        super().__init__(state_size, action_size, seed)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, state_size, seed)

        # Q-Network
        self.network_local = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.network_target = DuelingDQN(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        :param gamma: discount factor.
        """

        tree_idx, states, actions, rewards, next_states, dones, ISWeights = experiences

        # Get expected Q values from local model
        Q_expected = self.network_local(states).gather(1, actions)

        # Get next actions based on local network
        next_actions = self.network_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model based on local model next actions
        Q_targets_next = self.network_target(next_states).detach().gather(1, next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Update transition priorities
        self.memory.batch_update(tree_idx, np.ravel(np.abs(Q_targets.numpy())))

        # Compute loss
        loss = (torch.Tensor(ISWeights).float().to(DEVICE) * F.mse_loss(Q_expected, Q_targets)).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)


# # # -------------------------- Memory classes -------------------------- # # #
class ReplayBuffer:
    """
    Fixed-size memory buffer to store experience tuples.
    """

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer;
        :param batch_size: size of each training batch;
        :param seed: random seed.
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple(
            'Experience',
            field_names=('state', 'action', 'reward', 'next_state', 'done')
        )

        # initialize random number generator state
        random.seed(seed)

    def __len__(self):
        """
        Return the current size of internal memory.
        """

        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.

        :param state: state description;
        :param action: action taken in state;
        :param reward: reward received;
        :param next_state: next state;
        :param done: terminal state indicator.
        """

        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        :return: torch tensors of states, action, rewards, next states and terminal state flags.
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(DEVICE)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(DEVICE)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(DEVICE)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(DEVICE)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(DEVICE)

        return states, actions, rewards, next_states, dones


class SumTree:
    """
    This is slightly modified version of modified version of SumTree code of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py

    Credits to Thomas Simonini: https://github.com/simoninithomas
    """

    def __init__(self, capacity):
        """
        Create tree.

        :param capacity: number of tree leafs.
        """

        self.capacity = capacity
        self.data_pointer = 0

        # Generate the tree with all nodes values = 0
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

        self._override = False

    def __len__(self):
        return self.capacity if self._override else self.data_pointer

    def add(self, data, priority):
        """
        Add data and priority in the position of the data pointer.

        :param data: state transition data;
        :param priority: transition priority.
        """

        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # For __len__ magic method
        if self.data_pointer == 0:
            self._override = True

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index: int, priority):
        """
        Update the leaf priority and propagate the change through tree

        :param tree_index: corresponding leaf index in the tree
        :param priority: priority value
        """

        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Get leaf index, priority and experience associated with this leaf.

        :param v: sampling probability;
        :return: leaf_index, priority, experience tuple;
        """

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        """
        Return the root node of the tree.

        :return: root node of the tree.
        """

        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    This code is modified version of memory implementation by Thomas Simonini:
    https://github.com/simoninithomas
    """

    def __init__(self, buffer_size, batch_size, state_size, seed):
        # Making the tree
        self.tree = SumTree(buffer_size)

        self.batch_size = batch_size
        self.state_size = state_size

        # Hyperparameters
        self.min_priority = 0.01    # small number to avoid 0 probabilities when sampling
        self.alpha = 0.6            # sampling probability distribution adjustment
        self.beta = 0.3             # compensation for prioritized sampling bias

        self.beta_rise = 0.001      # rise of beta toward end of learning

        # clipped absolute error
        self.absolute_error_upper = 1

        np.random.seed(seed)

    def __len__(self):
        return len(self.tree)

    def push(self, state, action, reward, next_state, done):
        """
        Add new experience to the memory assigning the highest priority
        """

        # Find the maximum priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # Assemble experience array (S, A, R, S', done)
        experience = np.array([state, action, reward, next_state, done])

        # If the max priority = 0 we can't put at is, since this experience will never have a chance to be selected
        # So we use a maximal priority to store this transition
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(experience, max_priority)

    def sample(self):
        """
        Sample a batch of experiences based on priorities.
        Steps:
            - subdivide priority range into self.batch_size bins;
            - uniformly sample value from each bin;
            - search SumTree and retrieve experiences with corresponding values of priority;
            - calculate importance sampling (IS) for each sample;

        :return: indices in the tree, batch, IS weights
        """

        # Create a sample array that will contain the batch
        states_np = np.zeros((self.batch_size, self.state_size))
        actions_np = np.zeros((self.batch_size, 1))
        rewards_np = np.zeros((self.batch_size, 1))
        next_states_np = np.zeros((self.batch_size, self.state_size))
        dones_np = np.zeros((self.batch_size, 1))

        idx = np.empty(self.batch_size, dtype=np.int32)
        ISWeights = np.empty(self.batch_size, dtype=np.float32)

        # Calculate priority bins
        priority_bin_size = self.tree.total_priority / self.batch_size

        # Linearly increase beta every time sampling is performed
        self.beta = np.min([1., self.beta + self.beta_rise])

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (self.batch_size * p_min) ** (-self.beta) if p_min > 0 else 1

        for i in range(self.batch_size):
            # Bin range
            a, b = priority_bin_size * i, priority_bin_size * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            # P(j);
            # no alpha exponent, because priorities stored in the tree are already raised to the power alpha
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = ((N * P(i)) ** -b) / max w_i
            ISWeights[i] = np.power(self.batch_size * sampling_probabilities, -self.beta) / max_weight

            idx[i] = index
            states_np[i, :], actions_np[i, 0], rewards_np[i, 0], next_states_np[i, :], dones_np[i, :] = data

        # Reformat collected data to match agents input
        states = torch.from_numpy(states_np).float().to(DEVICE)
        actions = torch.from_numpy(actions_np).long().to(DEVICE)
        rewards = torch.from_numpy(rewards_np).float().to(DEVICE)
        next_states = torch.from_numpy(next_states_np).float().to(DEVICE)
        dones = torch.from_numpy(dones_np).float().to(DEVICE)

        return idx, states, actions, rewards, next_states, dones, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        """
        Update priorities on the tree.

        :param tree_idx: indices in the SumTree;
        :param abs_errors: absolute TD errors;
        """

        abs_errors += self.min_priority
        # Hubert loss can be used here as an alternative
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.alpha)

        for index, priority in zip(tree_idx, priorities):
            self.tree.update(index, priority)
