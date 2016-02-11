import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # accuracy charts
    ticks = (2,4,8,16,32,64,128)
    inds = np.arange(7)
    width = 0.35
    dense_accuracies = [0.192, 0.356, 0.672, 0.918, 0.94, 0.936, 0.938]
    sparse_accuracies = [0.192, 0.204, 0.682, 0.88, 0.87, 0.89, 0.832]
    plt.bar(inds, dense_accuracies, width, color='r')
    plt.bar(inds + width, sparse_accuracies, width, color='y')
    plt.ylabel("accuracy on heldout set")
    plt.xlabel("number of hidden units")
    plt.gca().set_xticks(inds + width)
    plt.gca().set_xticklabels(ticks)
    plt.savefig("accuracy.png")
    plt.close()
    dense_speeds = [181, 224, 290, 452, 798, 1527, 2869]
    sparse_speeds = [181, 203, 191, 200, 199, 225, 256]
    plt.bar(inds, dense_speeds, width, color='r')
    plt.bar(inds + width, sparse_speeds, width, color='y')
    plt.ylabel("seconds")
    plt.xlabel("number of hidden units")
    plt.gca().set_xticks(inds + width)
    plt.gca().set_xticklabels(ticks)
    plt.savefig("speed.png")
    plt.close()
