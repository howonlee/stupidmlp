Sparse MLP and justifications (or: Poking At Causation, part 3a / 3)
===

You can get most of the performance of a neural network with an approximation to backpropagation faster than O(|W|), meaning faster than the order of the number of members of the weight matrix, with sparsification. Here is the evidence.




A neural network's weights, after the random initialization and training, is composed of numbers of radically unequal magnitude. If you take a histogram of the absolute value of all the weights, you will see a remarkably heavy-tailed histogram. But that high skew also means that nearly all of the weights are useless, and can be seen to be useless almost immediately.

This means that you can get away with a remarkably stupid optimization, if you have a fast sparse outer product. Just kill the useless weights after a few "burn-in" iterations of SGD, and then go through with the much, much sparsified net and get the rest of the iterations done much quicker. You still get most of the representational power of all of those hidden units.

Typical points in the neural network weight attractor space have this property. I don't know if the attractor _itself_ has a similar, scaling or heavy-tailed structure, but that should be investigated. I found [something in word2vec representations](http://howonlee.github.io/2016/02/05/Fractal-20Wordvecs.html), at least.

If you need a more sophisticated way to put it, note that the directions in the high-dimensional weight space which have the highest Lyapunov exponent contribute the most to the Kolmogorov-Sinai information (which can be estimated as the sum of the positive Lyapunov exponents), and of course gradient descent can be thought of as a flow in weight space. That can be a sort of theoretical justification for being able to kill nearly all of the weights, although it isn't quite clear what Kolmogorov-Sinai entropy means to this system.

Alternatively, you can think of this as the end realization of the [Han Pool Tran Dally 2015](http://arxiv.org/abs/1506.02626) paper on the sparsification that can be had on L1 regularization on neural networks (although I don't do _any_ regularization, for simplicity). Presumably, you can avoid doing all that work if those weights don't matter. That lab's work is mostly on energy efficiency, but they also mention computational efficiency. I have probably mostly been scooped by them, but this is at least a nifty neat trick, I think, and is entirely in software. I don't even _have_ a GPU.

The next step, obviously, is to generalize to tensors and do this with RNNs proper, deeper nets, and to do it with GPU's. I probably won't pay for AWS GPU boxes, so RNNs it is. I may even get away without doing LSTM or anything like that, because the ability of RNN's to get away with avoiding vanishing and exploding gradients seems quite like the ability of systems at criticality to get away with long correlation distances. Let us see if that thought holds up.

Thanks to everyone I thanked on the [first installment of this thing](http://howonlee.github.io/2016/01/21/Poking-20At-20Causation1.html). I haven't written the second installment of this thing yet. I haven't done it with RNNs yet, so that is why this is installment 3a (3b will by the RNN's). Do not hesitate to contact me at hlee . howon at gmail if you have any thoughts or questions.
