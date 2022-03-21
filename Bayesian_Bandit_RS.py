# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# Prior probability: Beta, with parameters alpha_ and beta_
# Likelihood function = Bernoulli
# =========================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from tqdm import tqdm
from celluloid import Camera # getting the camera
from IPython.display import HTML # to show the animation in Jupyter
## ********************************************************
## Parameters
num_trials = 2000
true_CTR_values = [0.10, 0.15, 0.35, 0.40]
## ********************************************************
class Bandit:
  def __init__(self, p):
    self.true_CTR = p
    self.alpha_ = 1
    self.beta_ = 1
    self.N = 0 # for information only
  ## ****************************
  ## draw a realization (sample) from the "TRUE" distribution, ...
  # ... and see if the user clicks on it (returns True) or not (returns False)!
  def pull(self):
    return np.random.random() < self.true_CTR
  ## ****************************
  # sample from (current posterior) Beta distribution
  def sample(self):
    return np.random.beta(self.alpha_, self.beta_)
  ## ****************************
  # update the parameters of the posterior distribution
  # We update the parameters after observing each sample (i.e. online learning)
  def update(self, x):
    # becase the likelihood is Bernouli
    self.alpha_ += x
    self.beta_ += 1 - x
    # update the total count of samples as well
    self.N += 1
## ********************************************************
def experiment():
  bandits = [Bandit(p) for p in true_CTR_values]
  #
  sample_points = [int(t) for t in np.linspace(0, num_trials+1, num=min(num_trials, 20))]
  # sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
  rewards = np.zeros(num_trials)
  fig, ax = plt.subplots()  # creating my fig
  # ax.title("Bandit distributions after {} trials".format(trial))
  ax.set_ylabel("Distributions of CTR for different Bandits")
  ax.set_xlabel("values of the CTR")
  # z = ax.legend()
  # z.remove()
  list_colors_for_plot = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
  camera = Camera(fig)  # the camera gets the fig we'll plot
  for i in tqdm(range(num_trials)):
    # Thompson sampling
    j = np.argmax([b.sample() for b in bandits])
    #
    #### plot the posteriors
    if i in sample_points:
      # plot(bandits, i, fig, ax)
      # plot
      x_plot = np.linspace(0, 1, 200)
      for k in range(len(bandits)):
        b = bandits[k]
        y = beta.pdf(x_plot, b.alpha_, b.beta_)
        plt.plot(x_plot, y, color = list_colors_for_plot[k])
      #
      # Adding the legend:
      try:
        list_legends = []
        for b in bandits:
          legend_for_curr_bandit = "true CTR = {}, est. CTR = {}, #samples = {}/{}".format(round(b.true_CTR, 2), round((b.alpha_ - 1)/(b.N), 2), b.N, sum([u.N for u in bandits]))
          list_legends.append(legend_for_curr_bandit)
          #
        plt.legend(tuple(list_legends))
      except ZeroDivisionError: # one of the bandits does not have the samples
        pass
      #
      camera.snap()  # the camera takes a snapshot of the plot
    #
    x = bandits[j].pull()
    #
    # update rewards
    rewards[i] = x
    #
    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)
  animation = camera.animate()  # animation ready
  # HTML(animation.to_html5_video())  # displaying the animation
  animation.save("./figs/results.mp4")
  #
  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / num_trials)
  print("num times selected each bandit:", [b.N for b in bandits])
## ********************************************************
if __name__ == "__main__":
  experiment()
  H = 5+6

