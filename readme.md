Stupid Optimization on MLP (or: Poking At Causation, part 3a / 3)
===

TL;DR: You can get away with doing backpropagation much faster than you think with a ridiculously stupid trick: kill the small weights after a desultory burn-in period and do sparse outer products only on surviving weights.

There's quite a few multilayer perceptron pruning approaches, mostly for regularization. I tend to believe that most are not stupid enough. Regularization-based approaches, for example, don't make the actual optimization process any faster, and the brute-force and sensitivity-based approaches _cannot_ make the actual optimization faster because they happen after learning. Cascade correlation and the like are not feasible, because of local optima issues.

The reason why pruning _before_ learning would be important is that you can get much of the performance of a neural network with an approximation to backpropagation faster than O(|W|), meaning faster than the order of the number of members of the weight matrix, with sparsification. Here is the evidence, with a tinny little one-hidden-layer backpropagation multilayer perceptron. Note the x-axes.

![accuracy](http://i.imgur.com/yRzEZZz.png)

![speed](http://i.imgur.com/UyP8m2u.png)

Note that I only sparsified the input-hidden weight layer, assuming that its weights were going to dominate. This is less true for the 64 and 128-hidden unit layers. You could probably get better accuracy with L1 regularization. You can definitely get less ruthless with the sparsification: for example, you can get the accuracy back to 0.91 on MNIST by doing 64 hidden units and sparsifying to about 7000 params total only.

Why?
--

A neural network's weights, after the initialization but during and after training, is composed of numbers of radically unequal magnitude. If you take a histogram of the absolute value of all the weights, you will [see a remarkably heavy-tailed histogram](https://github.com/howonlee/mlp_gradient_histograms). But that high skew also means that nearly all of the weights are useless, and can be seen to be useless almost immediately. It also means that you should be suspicious of a positive feedback effect, which is indeed seen with the gradients.

I haven't found this cited in the literature after a long while searching, but the literature is sort of filled with [lots of articles](http://arxiv.org/abs/1506.02626) noting skew distributions in actual neurons in actual brains that you can squish, so that's probably worth mentioning.

Anyhow, the heavy tail phenomenon means that you can get away with a remarkably stupid optimization, if you have a sparse outer product operation. Just kill the comparatively-useless (lower magnitude) weights forever after a few "burn-in" iterations of SGD, and then go through with the much, much sparsified net using sparse outer products and get the rest of the iterations done much quicker. You still get most of the representational power of all of those hidden units, as you can see above.

Typical points in the neural network weight attractor space have this heavy-tail property, with a remarkable durability with respect to the actual dataset. I don't know if the attractor _itself_ has a similar, scaling or heavy-tailed structure, but that should be investigated. I found [something in word2vec representations](http://howonlee.github.io/2016/02/05/Fractal-20Wordvecs.html), at least.

If you need a more sophisticated reason, note that the directions in the high-dimensional weight space which have the highest Lyapunov exponent contribute the most to the [Kolmogorov-Sinai entropy](http://www.scholarpedia.org/article/Kolmogorov-Sinai_entropy) (which can be estimated as the sum of the positive Lyapunov exponents), and of course gradient descent can be thought of as a discrete flow in weight space. That can be a sort of theoretical justification for being able to kill nearly all of the weights. Although it isn't quite clear what Kolmogorov-Sinai entropy means to this system, it _should_ mean something, as the network obviously has more information as the movement in the gradient descent continues.

Alternatively, you can think of this as a much stupider realization of the [Han Pool Tran Dally 2015](http://arxiv.org/abs/1506.02626) paper, in that you can get away with doing all the pruning after very few iterations. Presumably, you can avoid doing all that work if those weights don't matter. (If I were a reviewer reading this thing as a paper, I would to be honest say that they scooped me)

The next step, obviously, is to do this with RNNs proper, deeper nets, and to do it with GPU's. I probably won't pay for AWS GPU boxes, so RNNs it is. I may even get away without doing LSTM or anything like that, because the ability of LSTM's to get away with avoiding vanishing and exploding gradients seems quite like the ability of systems at criticality to get away with long correlation distances. Let us see if that thought holds up.

Thanks to everyone I thanked on the [first installment of this thing](http://howonlee.github.io/2016/01/21/Poking-20At-20Causation1.html). I haven't written the second installment of this thing yet. I haven't done it with RNNs yet, so that is why this is installment 3a (3b will be the RNN's). Do not hesitate to contact me at hlee . howon at gmail if you have any thoughts or questions.

Running the Code
---

Download but don't unpack MNIST from here. I have code with CIFAR but I haven't been testing with it, but all you need to do with CIFAR is just download and unpack it into a directory named cifar-10-batches-py. Then just run `mlp.py`. The progress prints to stderr, the results to stdout, so I just pipe stdout to a file.
