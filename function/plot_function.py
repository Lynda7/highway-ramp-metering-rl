
import matplotlib.pyplot as plt


def plot_graphs(total_waiting_times, total_rewards, model_filename):
    # Plot for total waiting times
    plt.plot(range(len(total_waiting_times)), total_waiting_times)
    plt.xlabel("Epochs")
    plt.ylabel("Total Time")
    plt.title("The evolution of total waiting time over epochs")
    plt.savefig(f'visualizations/total_waiting_time_{model_filename}.png')
    plt.show()

    # Plot for total rewards
    plt.plot(range(len(total_rewards)), total_rewards)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Plot the rewards over epochs")
    plt.savefig(f'visualizations/scores_{model_filename}.png')
    plt.show()