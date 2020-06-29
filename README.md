# Navigation task using Deep Q Network
![](animation.gif)

### Introduction
The aim of this project is to develop an agent that is able to efficiently navigate in the large square world and collect bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.

- **`1`** - move backward.

- **`2`** - turn left.

- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

List of implemented solutions:

- Deep Q Network (DQN),

- Double DQN,

- Dueling DQN,

- Double Dueling DQN,

- Double Dueling DQN with prioritized experience replay.


### Getting Started

***Note: This guide is only valid for macOS.***

Please follow these steps to be able to run this project:

 1. Install build tools (such as C++ compiler and etc.) by installing Xcode and then Xcode command-line tools following [one of the various guides](https://macpaw.com/how-to/install-command-line-tools) .

 2. Install dependencies. It is highly recommended to install all dependencies in virtual environment (see [guide](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-Virtual-Environment.md)).

    - Install Unity ML-Agents Toolkit following instruction from [this page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) (official GitHub of Unity ML-Agents Toolkit). It is very likely that most of you will only need to install `mlagents` and `unityagents`  packages with the following command:
        ```shell
        pip install mlagents unityagents
        ```
        It is highly recommended to use Python not higher then 3.7, because `TesnsorFlow` (one of the dependency for `mlagents` is only compatible with Python 3.7).

    - Install PyTorch with
        ```shell
        pip insall torch torchvision
        ```
        Please see [official installation guide](https://pytorch.org/get-started/locally/#mac-installation) for more information.

4. Download Unity environment that matches you OS and put it in the root directory of this project:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. Run `main.py` from terminal with
    ```shell
    python main.py
    ```
    or in your IDE and follow the prompt to pick an agent to use.

4. Run `visualize.py` to see intelligent agent with
    ```shell
    python vizualize.py
    ```
   and follow the prompt to pick an agent to visualize.

5. See `Navigation.ipynb` for some basic demonstrations.


### Notes and issues
1. **Only 1 solver can be picked for a single run.** To ensure results reproducibility the environment is closed each time the training is complete. It is now impossible to immediately launch this environment after closing. Therefore, it is currently impossible to train all available agents in a single run.
