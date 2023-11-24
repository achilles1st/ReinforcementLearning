import numpy as np
import matplotlib.pyplot as plt

def plot_results_mountain_car(agent, savefig=False):
    train_metrics, test_metrics = agent.train_metrics, agent.test_metrics

    # set figure size
    fig, ax_list = plt.subplots(2, 2, figsize=(13, 8))
    ax_list[0,0].set_title('Training')
    ax_list[0,0].plot(train_metrics['reward'], lw=2, color='green')
    ax_list[0,0].set_ylabel('Acc. Reward')
    ax_list[1,0].plot(train_metrics['loss'], lw=2, color='red')
    ax_list[1,0].set_xlabel('Episode')
    ax_list[1,0].set_ylabel('Loss')
    ax_list[1,0].set_yscale('log')

    ax_list[0,1].set_title('Testing')
    ax_list[0,1].plot(test_metrics['reward'], lw=2, color='green')
    ax_list[0,1].set_ylabel('Acc. Reward')
    ax_list[0,1].axhline(np.mean(test_metrics['reward']),color='black',linestyle='dashed',lw=4)

    ax_list[1,1].plot(agent.test_metrics['steps_to_success'], lw=2, color='green')
    ax_list[1,1].set_xlabel('Episode')
    ax_list[1,1].set_ylabel('Number of steps to success')

    # put margin between plots
    plt.tight_layout(pad=2.0)

    if savefig:
        plt.savefig(f'MountainCar_{agent.model_type.value.replace(" ", "_")}_'
                    f'gamma_{agent.gamma}_alpha_{agent.alpha}_eps_decay_{agent.eps_decay}.pdf')
    else:
        plt.show()
