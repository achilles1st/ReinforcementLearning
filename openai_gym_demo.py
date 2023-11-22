#######################################################################
# Reinforcement Learning KU - WS23
# An OpenAI Gym tutorial with environment rendering
# Based on the basic tutorials from: https://www.gymlibrary.dev
#######################################################################
# Install the OpenAI Gym library as below to run this smoothly:
#
# pip install gym[all]==0.25.2
#
#######################################################################
import gym

# env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("Blackjack-v1", render_mode="human")
# env = gym.make("Taxi-v3", render_mode="human")
# env = gym.make("ALE/Breakout-v5", render_mode="human")
# env = gym.make("ALE/Tennis-v5", render_mode="human")
env = gym.make("ALE/Boxing-v5", render_mode="human")

reset = env.reset()

num_actions = env.action_space.n
print('Number of actions: ', num_actions)
print('Dimensionality of the observation space: ', env.observation_space)


for _ in range(10000):
    some_random_policy = env.action_space.sample()
    observation, reward, done, info = env.step(some_random_policy)

    if done:
        reset = env.reset()
env.close()
