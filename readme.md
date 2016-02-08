Boop-Expando MLP (or: Poking At Causation, part 3a / 3)
===

Read my [thoughts about causation](http://howonlee.github.io/2016/01/21/Poking-20At-20Causation1.html) to see the inspiration.

I repeat from Koenig and Moo's [great C++ text](http://www.amazon.com/Accelerated-C-Practical-Programming-Example/dp/020170353X) an anecdote about telescope lenses. It is said to be faster to grind a 3-inch lens and _then_ to grind a 6-inch lens than to start off grinding a 6-inch lens. This is because you get over the many pitfalls of the lens-grinding process when you can afford to make lots of mistakes. This is, basically, an algorithmic multilayer perceptron version of this. It's an approximation only, but it should be a good one.

Alternatively, you can think of this as the end realization of the [Han Pool Tran Dally 2015](http://arxiv.org/abs/1506.02626) paper on the sparsification that can be had on L1 reg on neural networks (although I don't do _any_ reg, for simplicity). Presumably, you can avoid doing all that work if those weights don't matter. That lab's work is mostly on energy efficiency, but they also mention computational efficiency. I have probably mostly been scooped by them, but it's at least a nifty neat trick, I think.

Algorithm
===

Start off with a backprop MLP with fully dense layers (just one in this, to keep it simple), but with weights in a sparse matrix data structure, and with a "starting hidden layer" of a really small size (less than log_2 (end hidden layer size)). The basic idea is that you will expand this. But first, you just do backprop normally through it. That's O(input * log_2 (end hidden layer)), times the number of data points, etc etc.

Then, kill half the weights, any weight less than the median. Because of the radical skew nature of the weights, killing that many does not matter. You will kill them permanently, they will stop existing for the network. Your backprops have to be sparse matrix multiplications and stuff now. But if you take advantage of that sparsity, your backprop step would take (1/2) * log_2 (end hidden layer) ops.

Now, "expando" the hidden layer to have 2 * the current number of hidden units. Then train again (which takes O(input * log_2 (end hidden layer)), still, because we killed half the weights). Then kill half the weights again. And so on, until you "expando" to the hidden unit size that you want.

Of course, you went through more epochs this way. O(log_2(end hidden layer)) times more passes for what would have been one pass in normal backprop. So a full account is that it takes O(input * log_2(end hidden layer) ** 2). Ain't that spiffy? Very similar to FastDTW.

Results
===

Conclusion
===

The next step, obviously, is to generalize to tensors and do this with RNN's, and to do it with GPU's. I can't do that second one, don't own GPU's and can't pay for AWS GPU boxes.

Thanks to everyone I thanked on the [first installment of this thing](http://howonlee.github.io/2016/01/21/Poking-20At-20Causation1.html). I haven't written the second installment of this thing yet.
