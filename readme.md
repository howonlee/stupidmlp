Expando MLP (or: Poking At Causation, part 3a / 3)
===

Read my [thoughts about causation](http://howonlee.github.io/2016/01/21/Poking-20At-20Causation1.html) to see the inspiration.

I repeat from Koenig and Moo's [great C++ text](http://www.amazon.com/Accelerated-C-Practical-Programming-Example/dp/020170353X) an anecdote about telescope lenses. It is said to be faster to grind a 3-inch lens and _then_ to grind a 6-inch lens than to start off grinding a 6-inch lens. This is because you get over the many pitfalls of the lense-grinding process when you can afford to make lots of mistakes. This is, basically, an algorithmic multilayer perceptron version of this.

Alternatively, you can think of this as the end realization of the [Han Pool Tran Dally 2015](http://arxiv.org/abs/1506.02626) paper on the sparsification that can be had on L1 normalization on neural networks. Presumably, you can avoid doing all that work. That lab's work is mostly on energy efficiency, but they also mention computational efficiency.

The next step, obviously, is to generalize to tensors and do this with RNN's, and to do it with GPU's. I can't do that second one, don't own GPU's and can't pay for AWS GPU boxes.
