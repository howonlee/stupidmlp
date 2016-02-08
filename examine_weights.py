import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights = np.load("weight_mat.npy")
    plt.hist(np.abs(weights).ravel(), bins=100)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    # killed, basically
    curr_weights = np.array([member for member in list(weights.ravel()) if member != 0])
    print np.median(curr_weights)
    # kprint np.ones_like(weights).sum()
    # kprint (np.abs(weights) < 0.0000000000001).sum()
