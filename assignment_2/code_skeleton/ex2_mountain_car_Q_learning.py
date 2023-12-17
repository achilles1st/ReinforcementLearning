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
    print(
        f"It {it:4d} | {mode:5s} | reward {reward_mean:5.1f} | loss {loss_mean:5.2f} | steps_to_success {steps_to_success}")


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
            self.num_hidden = 128

            self.Q_model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dimensions, self.num_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.num_hidden, self.num_actions)
            )

            # Weight initialization
            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

            self.Q_model.apply(init_weights)
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

        state = convert(state)
        if is_training and np.random.rand() < self.eps:
            action = torch.tensor([[np.random.choice(self.num_actions)]], dtype=torch.int64)  # Cast to int64
        else:
            with torch.no_grad():
                q_values = self.Q_model(state)
                action = torch.argmax(q_values, dim=1, keepdim=True).long()  # Cast to int64
        return action
        # - with probability eps return a random action
        # - otherwise find the action that maximizes Q
        # - During the testing phase, we don't need to compute the gradient!
        #   (Hint: use torch.no_grad()). The policy should return torch tensors.
        # - Also, during testing, pick actions deterministically.

    def compute_loss(self, state, action, reward, next_state, next_action, done):
        state = convert(state)
        next_state = convert(next_state)
        action = action.long()
        next_action = next_action.long()
        reward = torch.tensor(reward).float().view(1, 1)  # Cast to float
        done = torch.tensor(done).float().view(1, 1)  # Cast to float

        # Compute Q(s, a)
        q_value = self.Q_model(state).gather(1, action)

        # Compute Q(s', a') for the next state
        with torch.no_grad():
            next_q_values = self.Q_model(next_state)
            next_q_value = next_q_values.gather(1, next_action)

        # Compute the target Q-value: Q-learning update rule
        target_q = reward + self.gamma * (1 - done) * next_q_value

        # Convert target_q to float tensor explicitly
        target_q = target_q.float()

        # Compute the loss using the defined criterion
        loss = self.criterion(q_value, target_q)
        return loss

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

        position = state[0]
        velocity = state[1]

        # Modify the reward based on the position and velocity of the car
        reward = env_reward + (0.5 * (0.5 - position) + 0.05 * (0.02 - velocity ** 2))
        return reward

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

    agent.max_episode_length = 200  # reset max episode length for testing
    agent.test(render=render_on_test)

    plot_results_mountain_car(agent, savefig=savefig)


if __name__ == '__main__':
    eps = 1.0
    gamma = 0.95
    eps_decay = 0.99
    alpha = 0.1
    num_train_episodes = 1000
    max_episode_length = 1000

    model_type = ModelType.LINEAR  # Task b
    # Task c: model_type = ModelType.NEURAL_NET

    train_test_agent(model_type=model_type, gamma=gamma, alpha=alpha, eps=eps, eps_decay=eps_decay,
                     num_train_episodes=num_train_episodes, num_test_episodes=100,
                     max_episode_length=max_episode_length, savefig=True, render_on_test=False)
