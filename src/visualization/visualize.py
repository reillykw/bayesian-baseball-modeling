import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def metropolis_hastings(param, n_iters, proposal_dist):
    pass


def gibbs(params):
    pass


def unnormalized_posterior(likelihood, prior, data):
    return likelihood(n=data[1], p=data[2]).pmf(data[0]) * prior.pdf(data[2])


########################################################
# >>>>>>>>>> Set up Metropolis Hastings <<<<<<<<<<<<<< #
########################################################

alpha = 10
beta = 10

flips = 100
heads = 47

prior = stats.beta(alpha, beta)
likelihood = stats.binom

proposal = stats.norm
theta = 0.3
sigma = 0.1
iters = 5000
naccept = 0
trace = []
for i in range(iters):
    theta_i = proposal.rvs(loc=theta, scale=sigma)
    acceptance_ratio = min(
        1,
        (unnormalized_posterior(likelihood, prior, (heads, flips, theta_i)) /
         unnormalized_posterior(likelihood, prior, (heads, flips, theta)))
    )
    unif = np.random.rand()
    if unif < acceptance_ratio:
        theta = theta_i
        naccept += 1
    trace.append(theta)

print(f'Acceptance rate for MCMC {naccept / len(trace)}')

burn = 500
trace_x = trace[burn:]

plt.style.use('seaborn')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
fig.suptitle(
    f'Metropolis Hastings Iters={iters} with efficiency {naccept / len(trace):.1%}\n\n',
    fontsize=18
)
ax1.set_title('\n')
ax1.plot(range(len(trace_x)), trace_x)
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$\theta$')
ax2.set_title('\n')
ax2.hist(trace_x, edgecolor='k')
ax2.set_xlabel(r'$\theta$')
plt.savefig('metropolis_hastings.png')
plt.show()
