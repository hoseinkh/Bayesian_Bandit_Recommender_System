# Bayesian Bandit Recommender System

We implement a Bayesian Bandit Recommender System.

<br />

## Task:

We have several bandits that we do not know the success rate of any of them. We want to implement an algorithm that automatically balances the exploration-exploitation, and achieves finds the optimal bandit for us.

Here we assume that the prior is Beta distribution and the likelihood function is Bernoulli. Hence we have conjugate prior, and we can calculate the closed form formula for the posterior using the following formula ([Ref](https://en.wikipedia.org/wiki/Conjugate_prior)):

<p float="left">
  <img src="/figs/Conjug_prior_Beta_Bernoulli.png" width="450" />
</p>



---

### Codes & Results

The results are shown in the following:

<p float="left">
  <img src="/figs/Bayesian_Bandit_results.gif" width="450" />
</p>

As you can see the algorithm spends little experiments on the sub-optimal bandits, and the majority of the experiments are performed on the optimal bandit.

------

