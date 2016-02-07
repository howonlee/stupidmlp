import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights = np.load("weight_mat.npy")
    plt.hist(np.abs(weights).ravel(), bins=100)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.show()
    # killed, basically
    print np.ones_like(weights).sum()
    print (np.abs(weights) < 0.0000000000001).sum()
