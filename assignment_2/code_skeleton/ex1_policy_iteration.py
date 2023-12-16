# Run `pip install "gymnasium[toy-text]"` before running this code

import numpy as np
import gymnasium as gym
from frozen_lake_utils import plot_frozenlake_policy_iteration_results


class PolicyIteration:
    def __init__(self, render=False):
        self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
                            render_mode='human' if render else None).unwrapped
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n

        # self.policy[s, a] = probability of taking action a in state s
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions  # uniform policy

        # Extract MDP dynamics from gym environment

        # self.P[s', s, a] = P(s' | s, a)
        self.P = np.zeros((self.num_states, self.num_states, self.num_actions))

        # self.r[s, a] = expected immediate reward in state s when taking action a
        self.r = np.zeros((self.num_states, self.num_actions))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    self.r[state, action] += prob * reward
                    self.P[next_state, state, action] += prob

    def policy_evaluation(self, gamma=1., theta=1e-8):
        """
        Run iterative policy evaluation for the current policy `self.policy`
        and return the corresponding state-value function `v`.
        :param gamma: discount factor
        :param theta: stop evaluation if delta < theta
        :return:
        """

        # P_pi[s', s] = P(s' | s) when acting according to self.policy
        P_pi = np.sum(self.P * self.policy, axis=-1)

        # r_pi[s] = expected immediate reward in state s when acting according to self.policy
        r_pi = np.sum(self.r * self.policy, axis=-1)

        V = np.zeros(self.num_states)
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(self.num_states):
                old_v = V[s]
                V[s] = r_pi[s] + gamma * np.sum(P_pi[:, s] * V)

                # How much our value function changed (across any states)
                delta = max(delta, np.abs(old_v - V[s]))


            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return V

    def compute_Q_from_v(self, v, gamma=1.):
        Q = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                # Calculate Q-value for each action in each state according to VO slides
                Q[s, a] = np.sum(self.P[:, s, a] * v) * gamma + self.r[s, a]

        assert Q.shape == (self.num_states, self.num_actions)
        return Q

    def policy_improvement(self, gamma=1.):
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions  # Start with a uniform policy

        while True:
            policy_old = self.policy.copy()
            v = self.policy_evaluation(gamma)
            Q = self.compute_Q_from_v(v, gamma)

            # Improve policy by acting greedily w.r.t. Q
            for s in range(self.num_states):
                best_actions = np.argwhere(Q[s] == np.max(Q[s])).flatten()  # Get indices of best actions
                updated_policy = np.zeros(self.num_actions)
                updated_policy[best_actions] = 1.0 / len(best_actions)  # Update probabilities for best actions
                self.policy[s] = updated_policy

            if np.array_equal(policy_old, self.policy):
                break

        return self.policy, v, Q

    def run_episode(self, max_episode_length=1000):
        """
        Run an episode with the current policy `self.policy`
        and return the sum of rewards.
        We stop the episode if we reach a terminal state or
        if the episode length exceeds `max_episode_length`.
        :param max_episode_length: maximum episode length
        :return: sum of rewards of the episode
        """

        episode_reward = 0
        state, _ = self.env.reset()
        # sample action out of policy
        action = np.random.choice(self.num_actions, p=self.policy[state])

        for t in range(max_episode_length):
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            next_action = np.random.choice(self.num_actions, p=self.policy[next_state])
            state, action = next_state, next_action
            if done:
                break

        return episode_reward

    def test_policy(self, num_episodes):
        """
        Run `num_episodes` episodes with the current policy `self.policy`.
        :param num_episodes: number of episodes to run
        :return: list of rewards of the episodes
        """

        rewards = []
        for _ in range(num_episodes):

            rewards.append(self.run_episode())

        return rewards


if __name__ == '__main__':
    # If you want to see the agent in action, set render=True
    policy_iteration = PolicyIteration(render=False)
    for gamma in [0.95, 1.0]:
        pi_star, v_star, q_star = policy_iteration.policy_improvement(gamma=gamma)
        test_rewards = policy_iteration.test_policy(num_episodes=1000)
        plot_frozenlake_policy_iteration_results(policy_iteration, gamma, test_rewards,
                                                 v_star, q_star, savefig=True)