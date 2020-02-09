# Assignment Statement

The assignment is the same as the solved example with _**10-armed bandits**_ except that you should also implement **UCB exploration** in addition to **epsilon-greedy policy**. 

- Try a few different values of epsilon -- say 0.5, 0.3, 0.2, 0.1, 0.05, 0.01 for epsilon-greedy
- For UCB scheme, try parameter c values of say 5, 3, 2, 1, 0.5, 0.1.
- Also, in the book, experiments are shown for the case when mean rewards is picked from `N(0,1)`. Then, the arm _i_ actual reward is generated using `N(mean-i, 1)`, i.e., variance is **1**.
Please repeat this whole experiment when the variance of reward is **10** instead of **1** , i.e., `N(0,10)` is used for generating mean rewards for the 10 arms, and `N(mean-i, 10)` for generating individual rewards for each arm, and study the effect of epsilon in the epsilon greedy scheme and parameter c in the UCB scheme.

_**Note**_:
The standard normal `N(0,1)` random variable code is already available on internet.
- To generate `N(c,1)` from `N(0,1)` add **c** to the `N(0,1)` sample.
- To get `N(0,10)` from `N(0,10)` sample, multiply by `sqrt{10}` (square root of 10), the sample value obtained from `N(0,1)`.
