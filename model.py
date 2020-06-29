import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size: int, action_size: int, seed: int,
                 fc1_units: int = 256, fc2_units: int = 256, fc3_units: int = 128):
        """
        Initialize parameters and build model.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed;
        :param fc1_units: number of nodes in first hidden layer;
        :param fc2_units: number of nodes in second hidden layer;
        :param fc3_units: number of nodes in second hidden layer.
        """

        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """
        Build a network that maps state -> action values.

        :param state: state tensor;
        :return: action values.
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingDQN(nn.Module):
    """
    Actor (Policy) Model with two stream (Double DQN).
    """

    def __init__(self, state_size: int, action_size: int, seed: int,
                 fc1_units: int = 256, fc2_units: int = 256, fc3_units: int = 128):
        """
        Initialize parameters and build model.

        :param state_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed;
        :param fc1_units: number of nodes in first hidden layer;
        :param fc2_units: number of nodes in second hidden layer;
        :param fc3_units: number of nodes in second hidden layer.
        """

        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.value_stream = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size)
        )

    def forward(self, state):
        """
        Build a network that maps state -> action values.

        :param state: state tensor;
        :return: action values.
        """

        state_values = self.value_stream(state)
        advantages = self.advantage_stream(state)
        return state_values + (advantages - advantages.mean())
