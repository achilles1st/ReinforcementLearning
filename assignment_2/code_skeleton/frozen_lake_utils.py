import matplotlib.pyplot as plt
import numpy as np


def plot_results(range1, range2, range1_label, range2_label, train_results, test_results):
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, sharex='all', sharey='all',
                             figsize=(6.97, 6.97 / 1.618))
    X, Y = np.meshgrid(range1, range2, indexing='ij')
    for ax, Z, title in zip(axes, [train_results, test_results], ['training phase', 'test phase']):
        ax.plot_surface(X, Y, Z.mean(-1), cmap='coolwarm')
        ax.set_title(title)
        ax.set_xlabel(range1_label)
        ax.set_ylabel(range2_label)
        ax.set_zlabel('reward')

    plt.suptitle('Deterministic SARSA')
    plt.show()


def plot_value_function(ax, v, is_v_star=False):
    v = v.reshape(4, 4)
    ax.imshow(v, interpolation='nearest', cmap='coolwarm', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])

    # Loop over data dimensions and create text annotations.
    for i in range(4):
        for j in range(4):
            ax.text(j, i, np.round(v[i, j], 3), ha="center", va="center", color="w")

    ax.set_title('Value function ' + ('$v^*(s)$' if is_v_star else '$v_\pi(s)$'))


def plot_policy(ax, env, Q_table):
    n_state, n_action = Q_table.shape

    def to_map(c):
        if c == b'S':
            return 2
        elif c == b'F':
            return 1
        elif c == b'H':
            return -2
        else:
            return 0

    map0 = np.vectorize(to_map)(env.desc)

    def check_terminal(c):
        if c in [b'F', b'S']:
            return 0
        else:
            return 1

    is_terminal = np.vectorize(check_terminal)(env.desc)

    start_position = np.concatenate(np.where(env.desc == b'S'))
    goal_position = np.concatenate(np.where(env.desc == b'G'))
    assert start_position.shape == (
        2,), 'Weird start position {}'.format(goal_position)
    assert goal_position.shape == (
        2,), 'Weird goal position {}'.format(goal_position)

    is_terminal = np.array(is_terminal, dtype=bool)
    map0 = np.array(map0)

    ax.pcolormesh(np.arange(env.ncol + 1), np.arange(env.nrow + 1), map0, cmap='coolwarm')

    def next_state(i, j, a):
        if a == 0:
            return i, j - 1
        elif a == 1:
            return i + 1, j
        elif a == 2:
            return i, j + 1
        elif a == 3:
            return i - 1, j
        else:
            raise ValueError('Unkown action {}'.format(a))

    ax.set_ylim([env.ncol, 0])

    for s in range(n_state):
        i, j = s // env.nrow, np.mod(s, env.ncol)
        if not (is_terminal[i, j]):
            a = np.argmax(Q_table[s, :])
            i_, j_ = next_state(i, j, a)
            scale = 0.8
            if (j + 0.5) + (j_ - j) * scale < 0:
                scale = 0.3
            ax.arrow(j + .5,
                     i + .5, (j_ - j) * scale, (i_ - i) * scale,
                     head_width=.1,
                     color='white')

    ax.annotate('S',
                xy=(start_position[0] + .35, start_position[1] + .65),
                color='black',
                size=20)
    ax.annotate('G',
                xy=(goal_position[0] + .35, goal_position[1] + .65),
                color='black',
                size=20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Policy')


def plot_frozenlake_policy_iteration_results(policy_iteration, gamma, test_rewards, v_star, q_star, savefig=False):
    plot_frozenlake_results('Policy Iteration', q_star, policy_iteration.env, gamma, test_rewards,
                            v=v_star, savefig=savefig)

def plot_frozenlake_model_free_results(agent, gamma, savefig=False):
    plot_frozenlake_results(agent.algorithm.value, agent.Q, agent.env, gamma, agent.test_reward,
                            agent.train_reward, savefig=savefig)

def plot_frozenlake_results(algo_name, Q_table, env, gamma, test_reward, train_reward=None, v=None, savefig=False):
    n_state, n_action = Q_table.shape
    grid_size = np.sqrt(n_state)

    fig, ax_list = plt.subplots(2, 2)
    # set suptitle
    fig.suptitle(f'FrozenLake | {algo_name} with $\gamma={gamma}$')

    if train_reward:
        train_reward = np.array(train_reward)
        un_r = np.unique(train_reward)
        assert (un_r == [0, 1]).all() or (un_r == [0]).all() or (un_r == [1]).all(), '''
        The reward list should be only 0 and 1 for the FrozenLake problem.
        Now we found {}'''.format(np.unique(train_reward))

        # plot the reward over time
        ax_list[0, 0].plot(train_reward, lw=2)
    else:
        # write text "No training needed" in the plot
        ax_list[0, 0].text(0.5, 0.5, 'No training needed\n\n(Policy Iteration)',
                           horizontalalignment='center', verticalalignment='center', fontsize=10)

    ax_list[0, 0].set_ylim([0, 1.1])
    ax_list[0, 0].set_title('Training')
    ax_list[0, 0].set_ylabel('Accumulated reward')
    ax_list[0, 0].set_xlabel('Trial number')

    test_reward = np.array(test_reward)

    # plot the reward over time
    r_av = np.mean(test_reward)
    ax_list[0, 1].plot(test_reward, lw=2)
    ax_list[0, 1].set_ylim([0, 1.1])
    ax_list[0, 1].axhline(y=r_av, color='red', lw=1)
    ax_list[0, 1].set_yticks([0, r_av, 1])
    ax_list[0, 1].set_ylim([0, 1.1])
    ax_list[0, 1].set_title('Testing')
    ax_list[0, 1].set_xlabel('Trial number')

    # plot the policy
    plot_policy(ax_list[1, 0], env, Q_table)
    is_v_star = v is not None
    if v is None:
        v = np.max(Q_table, axis=1)

    plot_value_function(ax_list[1, 1], v, is_v_star)

    plt.tight_layout()
    if savefig:
        plt.savefig(f'FrozenLake_{algo_name.replace(" ", "_")}_gamma_{gamma}.pdf')
    else:
        plt.show()