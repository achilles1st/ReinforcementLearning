# Run `pip install "gymnasium[toy-text]"` before running this code

import numpy as np
import gymnasium as gym
from frozen_lake_utils import plot_frozenlake_model_free_results
from enum import Enum

class RLAlgorithm(Enum):
    SARSA = 'SARSA'
    Q_LEARNING = 'Q-Learning'
    EXPECTED_SARSA = 'Expected SARSA'

class ModelFreeAgent:
    def __init__(self, algorithm, alpha, eps, gamma, eps_decay,
                 num_train_episodes, num_test_episodes, max_episode_length):
        self.algorithm = algorithm
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.num_train_episodes = num_train_episodes
        self.num_test_episodes = num_test_episodes
        self.max_episode_length = max_episode_length
        self.test_reward, self.train_reward = None, None

        self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                            is_slippery=True, render_mode='human').unwrapped
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n

        self.Q = np.zeros((self.num_states, self.num_actions))

    def set_render_mode(self, render: bool):
        self.env.render_mode = 'human' if render else None

    def policy(self, state, is_training):
        """
        Given a state, return an action according to an epsilon-greedy policy.
        :param state: The current state
        :param is_training: Whether we are training or testing the agent
        :return: An action (int)
        """

        if is_training and np.random.rand() < self.eps:
            # Exploration: Choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: Choose the action that maximizes Q
            return np.argmax(self.Q[state])

        # - with probability self.eps return a random action
        # - otherwise find the action that maximizes self.Q
        # - when testing, do not use epsilon-greedy exploration but always return the greedy action


    def train_step(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table `self.Q` according to the used algorithm (self.algorithm).

        :param state: State we were in *before* taking the action
        :param action: Action we have just taken
        :param reward: Immediate reward received after taking the action
        :param next_state: State we are in *after* taking the action
        :param next_action: Next action we *will* take (sampled from our policy)
        :param done: If True, the episode is over
        """

        if self.algorithm == RLAlgorithm.SARSA:
            next_q = self.Q[state, action]
            target = reward + self.gamma * self.Q[next_state, next_action]
            self.Q[state, action] += self.alpha * (target - next_q)
            # - Q(s, a) = alpha * (reward + gamma * Q(s', a') - Q(s, a))

        elif self.algorithm == RLAlgorithm.Q_LEARNING:
            max_next_q = np.max(self.Q[next_state])
            target = reward + self.gamma * max_next_q
            self.Q[state][action] += self.alpha * (target - self.Q[state][action])

            # - Q(s, a) = alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
            # - where the max is taken over all possible actions

        elif self.algorithm == RLAlgorithm.EXPECTED_SARSA:
            # in order to want to use the greedy policy to select the best action without exploration, hence is_training=False
            next_q = self.Q[state, action]

            expected_q = 0
            q_max = np.max(self.Q[next_state, :])
            greedy_actions = 0
            for i in range(self.num_actions):
                if self.Q[next_state][i] == q_max:
                    greedy_actions += 1

            non_greedy_action_probability = self.eps / self.num_actions
            greedy_action_probability = ((1 - self.eps) / greedy_actions) + non_greedy_action_probability

            for i in range(self.num_actions):
                if self.Q[next_state][i] == q_max:
                    expected_q += self.Q[next_state][i] * greedy_action_probability
                else:
                    expected_q += self.Q[next_state][i] * non_greedy_action_probability

            target = reward + self.gamma * expected_q
            self.Q[state, action] += self.alpha * (target - next_q)

            # - Q(s, a) = alpha * (reward + gamma * E[Q(s', a')] - Q(s, a))
            # - where the expectation E[Q(s', a')] is taken wrt. actions a' of the policy (s' is given by next_state)

    def run_episode(self, training, render=False):
        """
        Run an episode with the current policy `self.policy`
        and return the sum of rewards.
        We stop the episode if we reach a terminal state or
        if the episode length exceeds `self.max_episode_length`.
        :param training: True if we are training the agent, False if we are testing it
        :param render: True if we want to render the environment, False otherwise
        :return: sum of rewards of the episode
        """

        self.set_render_mode(render)

        episode_reward = 0
        state, _ = self.env.reset()
        action = self.policy(state, training)
        for t in range(self.max_episode_length):
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            next_action = self.policy(next_state, training)
            if training:
                self.train_step(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            if done:
                break

        return episode_reward

    def train(self):
        """
        Train the agent for self.max_train_iterations episodes.
        After each episode, we decay the exploration rate self.eps using self.eps_decay.
        After training, self.train_reward contains the reward-sum of each episode.
        """

        self.train_reward = []
        for _ in range(self.num_train_episodes):
            self.train_reward.append(self.run_episode(training=True))
            self.eps *= self.eps_decay

    def test(self, render=False):
        """
        Test the agent for `num_episodes` episodes.
        After testing, self.test_reward contains the reward-sum of each episode.
        :param num_episodes: The number of episodes to test the agent
        :param render: True if we want to render the environment, False otherwise
        """

        self.test_reward = []
        for _ in range(self.num_test_episodes):
            self.test_reward.append(self.run_episode(training=False, render=render))


def train_test_agent(algorithm, gamma, alpha, eps, eps_decay,
                     num_train_episodes=10_000, num_test_episodes=5000,
                     max_episode_length=200, render_on_test=False, savefig=True):
    """
    Trains and tests an agent with the given parameters.

    :param algorithm: The RLAgorithm to use (SARSA, Q_LEARNING, EXPECTED_SARSA)
    :param gamma: Discount rate
    :param alpha: "Learning rate"
    :param eps: Initial exploration rate
    :param eps_decay: Exploration rate decay
    :param num_train_episodes: Number of episodes to train the agent
    :param num_test_episodes: Number of episodes to test the agent
    :param max_episode_length: Episodes are terminated after this many steps
    :param render_on_test: If true, the environment is rendered during testing
    :param savefig: If True, saves a plot of the result figure in the current directory. Otherwise, we show the plot.
    :return:
    """

    agent = ModelFreeAgent(algorithm=algorithm, alpha=alpha, eps=eps,
                           gamma=gamma, eps_decay=eps_decay,
                           num_train_episodes=num_train_episodes,
                           num_test_episodes=num_test_episodes,
                           max_episode_length=max_episode_length)
    agent.train()
    agent.test(render=render_on_test)
    plot_frozenlake_model_free_results(agent, gamma, savefig=savefig)
    print(f'{algorithm.value}  Mean Training Reward: {np.mean(agent.train_reward)} |',
          f'Mean Test Reward: {np.mean(agent.test_reward)} | {gamma=}, {alpha=}, {eps=}, {eps_decay=}')


if __name__ == '__main__':
    eps = 1
    alpha_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
    eps_decay_list = [0.99, 0.995, 0.999, 0.9995, 0.9999]

    for algo in [RLAlgorithm.SARSA, RLAlgorithm.Q_LEARNING, RLAlgorithm.EXPECTED_SARSA]:
        for gamma in [0.95, 1]:
            for alpha in alpha_list:
                for eps_decay in eps_decay_list:
                    train_test_agent(algorithm=algo, gamma=gamma, alpha=alpha, eps=eps, eps_decay=eps_decay,
                                     num_train_episodes=10_000, num_test_episodes=5_000,
                                     max_episode_length=200, savefig=False)