import numpy as np
import gymnasium as gym
from mountain_car_utils import plot_results_mountain_car
from enum import Enum
import torch

def convert(x):
    return torch.tensor(x).float().unsqueeze(0)

def update_metrics(metrics, episode):
    for k, v in episode.items():
        metrics[k].append(v)

def print_metrics(it, metrics, is_training, window=50):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    steps_to_success = np.mean(metrics['steps_to_success'][-window:])
    print(f"It {it:4d} | {mode:5s} | reward {reward_mean:5.1f} | loss {loss_mean:5.2f} | steps_to_success {steps_to_success}")

class ModelType(Enum):
    LINEAR = 'Linear'
    NEURAL_NET = 'Neural Network'

class QLearningMountainCarAgent:
    def __init__(self, model_type, alpha, eps, gamma, eps_decay,
                 num_train_episodes, num_test_episodes, max_episode_length):
        self.model_type = model_type
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.num_train_episodes = num_train_episodes
        self.num_test_episodes = num_test_episodes
        self.max_episode_length = max_episode_length
        self.test_metrics, self.train_metrics = None, None

        self.env = gym.make('MountainCar-v0', render_mode='human').unwrapped
        self.num_actions = self.env.action_space.n
        self.state_dimensions = self.env.observation_space.shape[0]

        if model_type == ModelType.LINEAR:
            self.Q_model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dimensions, self.num_actions, bias=False)
            )
        elif model_type == ModelType.NEURAL_NET:
            # TODO: Implement a neural network with one hidden layer which consists of `self.num_hidden` neurons.
            self.num_hidden = None
            raise NotImplementedError('NeuralNet not implemented')

        self.optimizer = torch.optim.Adam(self.Q_model.parameters(), lr=self.alpha)
        self.criterion = torch.nn.MSELoss()

    def set_render_mode(self, render: bool):
        self.env.render_mode = 'human' if render else None

    def policy(self, state, is_training):
        """
        Given a state, return an action according to an epsilon-greedy policy.
        :param state: The current state
        :param is_training: Whether we are training or testing the agent
        :return: An action (torch.tensor)
        """

        # TODO: Implement an epsilon-greedy policy
        # - with probability eps return a random action
        # - otherwise find the action that maximizes Q
        # - During the testing phase, we don't need to compute the gradient!
        #   (Hint: use torch.no_grad()). The policy should return torch tensors.
        # - Also, during testing, pick actions deterministically.

    def compute_loss(self, state, action, reward, next_state, next_action, done):
        """
        Compute the loss, defined to be the MSE between
        Q(s,a) and the Q-Learning target y.

        :param state: State we were in *before* taking the action
        :param action: Action we have just taken
        :param reward: Immediate reward received after taking the action
        :param next_state: State we are in *after* taking the action
        :param next_action: Next action we *will* take (sampled from our policy)
        :param done: If True, the episode is over
        """
        state = convert(state)
        next_state = convert(next_state)
        action = action.view(1, 1)
        next_action = next_action.view(1, 1)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).int().view(1, 1)

        # TODO: Compute Q(s, a) and Q(s', a') for the given state-action pairs.
        # Detach the gradient of Q(s', a'). Why do we have to do that? Think about
        # the effect of backpropagating through Q(s, a) and Q(s', a') at once!

        # TODO: Return the loss computed using self.criterion.

    def train_step(self, state, action, reward, next_state, next_action, done):
        """
        Perform an optimization step on the loss and return the loss value.
        """

        loss = self.compute_loss(state, action, reward, next_state, next_action, done)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def custom_reward(self, env_reward, state):
        """
        Compute a custom reward for the given environment reward and state.
        :param env_reward: Reward received from the environment
        :param state: State we are now in (after playing action)
        :return: Custom reward value (float)
        """

        # TODO: Implement a custom reward function
        # Right now, we just return the environment reward.
        return env_reward

    def run_episode(self, training, render=False):
        """
        Run an episode with the current policy `self.policy`
        and return a dictionary with metrics.
        We stop the episode if we reach the goal state or
        if the episode length exceeds `self.max_episode_length`.
        :param training: True if we are training the agent, False if we are testing it
        :param render: True if we want to render the environment, False otherwise
        :return: sum of rewards of the episode
        """

        self.set_render_mode(render)

        steps_to_success, episode_loss, episode_reward = -1, 0, 0
        state, _ = self.env.reset()
        action = self.policy(state, training)
        for t in range(self.max_episode_length):
            next_state, env_reward, done, _, _ = self.env.step(action.item())
            reward = self.custom_reward(env_reward, next_state)
            episode_reward += reward
            next_action = self.policy(next_state, training)
            if training:
                episode_loss += self.train_step(state, action, reward, next_state, next_action, done)
            else:
                with torch.no_grad():
                    episode_loss += self.compute_loss(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
            if done:
                if t < (self.max_episode_length - 1):
                    steps_to_success = t

                break

        # return episode_reward
        return dict(reward=episode_reward, loss=episode_loss / t, steps_to_success=steps_to_success)


    def train(self):
        """
        Train the agent for self.max_train_iterations episodes.
        After each episode, we decay the exploration rate self.eps using self.eps_decay.
        After training, self.train_reward contains the reward-sum of each episode.
        """

        self.Q_model.train()
        self.train_metrics = dict(reward=[], loss=[], steps_to_success=[])
        for it in range(self.num_train_episodes):
            episode_metrics = self.run_episode(training=True, render=False)
            update_metrics(self.train_metrics, episode_metrics)
            print_metrics(it, self.train_metrics, is_training=True)
            self.eps *= self.eps_decay

    def test(self, render=False):
        """
        Test the agent for `self.num_test_episodes` episodes.
        After testing, self.test_metrics contains metrics of each episode.
        :param num_episodes: The number of episodes to test the agent
        :param render: True if we want to render the environment, False otherwise
        """

        self.Q_model.eval()
        self.test_metrics = dict(reward=[], loss=[], steps_to_success=[])
        for it in range(self.num_test_episodes):
            episode_metrics = self.run_episode(training=False, render=render)
            update_metrics(self.test_metrics, episode_metrics)
            print_metrics(it, self.test_metrics, is_training=False)


def train_test_agent(model_type, gamma, alpha, eps, eps_decay,
                     num_train_episodes=2000, num_test_episodes=100,
                     max_episode_length=200, render_on_test=False, savefig=True):
    """
    Trains and tests an agent with the given parameters.

    :param model_type: The type of model to use (linear or neural network)
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

    agent = QLearningMountainCarAgent(model_type, alpha=alpha, eps=eps,
                                      gamma=gamma, eps_decay=eps_decay,
                                      num_train_episodes=num_train_episodes,
                                      num_test_episodes=num_test_episodes,
                                      max_episode_length=max_episode_length)
    agent.train()

    agent.max_episode_length = 200 # reset max episode length for testing
    agent.test(render=render_on_test)

    plot_results_mountain_car(agent, savefig=savefig)

if __name__ == '__main__':
    eps = 1.0
    gamma = None
    eps_decay = None
    alpha = None
    num_train_episodes = None
    max_episode_length = None

    model_type = ModelType.LINEAR # Task b
    # Task c: model_type = ModelType.NEURAL_NET

    train_test_agent(model_type=model_type, gamma=gamma, alpha=alpha, eps=eps, eps_decay=eps_decay,
                     num_train_episodes=num_train_episodes, num_test_episodes=100,
                     max_episode_length=max_episode_length, savefig=True, render_on_test=False)
