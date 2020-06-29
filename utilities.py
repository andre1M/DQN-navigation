from matplotlib import pyplot as plt
import numpy as np

from collections import deque


def train(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.1, eps_decay=0.995, decline='lin'):
    """
    Deep Q-Learning.
    Linear epsilon decline.

    :param agent: agent object to be trained;
    :param env: environment callable;
    :param n_episodes: maximum number of training episodes;
    :param eps_start: starting value of epsilon, for epsilon-greedy action selection;
    :param eps_end: minimum value of epsilon;
    :param eps_decay: multiplicative factor (per episode) for decreasing epsilon;
    :param decline: 'lin' for linear epsilon decline and 'exp' for exponential;
    :return: scores per episode.
    """

    allowed_declines = ['lin', 'exp']
    if decline not in allowed_declines:
        raise ValueError(f"Invalid epsilon decline mode '{decline}'.\n"
                         f"Allowed decline modes {allowed_declines}.")

    brain_name = env.brain_names[0]

    scores = []                         # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    eps = eps_start                     # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # reset score for new episode

        while True:
            action = agent.act(state, eps)                          # select action
            env_info = env.step(action)[brain_name]                 # get environment response to the action
            next_state = env_info.vector_observations[0]            # get the next state
            reward = env_info.rewards[0]                            # get the reward
            done = env_info.local_done[0]                           # terminal state flag
            agent.step(state, action, reward, next_state, done)     # process experience
            state = next_state
            score += reward
            if done:
                break

        # save recent scores
        scores_window.append(score)
        scores.append(score)

        # decrease epsilon
        if decline == allowed_declines[0]:
            eps = max(eps_end, eps - eps_decay)
        elif decline == allowed_declines[1]:
            eps = max(eps_end, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores


def plot_scores(scores, filename):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(np.arange(len(scores)), scores)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Episode #', fontweight='bold')
    ax.set_title('Score evolution over training', fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
