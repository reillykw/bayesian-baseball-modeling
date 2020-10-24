import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use('seaborn')

# alpha, beta = 10, 10
# prior = stats.beta(alpha, beta)
#
# n = 100
# h = 35
#
# likelihood = stats.binom
# posterior = stats.beta(h + alpha, n - h + beta)
#
# thetas = np.linspace(0, 1, 100)

# fig, ax = plt.subplots()
# ax.plot(thetas, prior.pdf(thetas), c='r', label='Prior')
# ax.fill_between(thetas, prior.pdf(thetas), facecolor='r', alpha=0.3)
# ax.plot(thetas, len(thetas) * likelihood(n, thetas).pmf(h), c='b', label='Likelihood')
# ax.fill_between(thetas, len(thetas) * likelihood(n, thetas).pmf(h), facecolor='b', alpha=0.3)
# ax.plot(thetas, posterior.pdf(thetas), c='g', label='Posterior')
# ax.fill_between(thetas, posterior.pdf(thetas), facecolor='g', alpha=0.3)
# plt.legend()
# plt.show()
#
#
# fig, axes = plt.subplots(1, 3)
# axes[0].plot(thetas, prior.pdf(thetas), c='r')
# axes[0].fill_between(thetas, prior.pdf(thetas), facecolor='r', alpha=0.3)
# axes[0].set_title('Prior')
# axes[1].plot(thetas, len(thetas) * likelihood(n, thetas).pmf(h), c='b')
# axes[1].fill_between(thetas, len(thetas) * likelihood(n, thetas).pmf(h), facecolor='b', alpha=0.3)
# axes[0].set_title('Likelihood')
# axes[2].plot(thetas, posterior.pdf(thetas), c='g')
# axes[2].fill_between(thetas, posterior.pdf(thetas), facecolor='g', alpha=0.3)
# axes[2].set_title('Posterior')
# plt.legend()
# plt.show()


thetas = np.linspace(0, 1, 11)


def plot_discrete_prior(ax, thetas):
    dens = np.minimum(thetas, 1 - thetas)
    dens /= np.sum(dens)
    ax.scatter(thetas, dens, c='b')
    for i, theta in enumerate(thetas):
        axes[0].axvline(x=theta, ymax=(dens[i] / 0.25), c='b', alpha=0.5)
    ax.set_ylim((0, 0.25))
    ax.set_title('Prior', fontweight='bold')
    ax.set_ylabel(r'$p(\theta)$')
    ax.set_xlabel(r'$\theta$')


def plot_discrete_likelihood(ax, thetas):
    ax.scatter(thetas, thetas, c='b')
    for i, theta in enumerate(thetas):
        ax.axvline(x=theta, ymax=theta, c='b', alpha=0.5)
    ax.set_ylim((0, 1))
    ax.set_title('Likelihood', fontweight='bold')
    ax.set_ylabel(r'$p(X|\theta)$')
    ax.set_xlabel(r'$\theta$')


def plot_discrete_posterior(ax, thetas):
    dens = np.minimum(thetas, 1 - thetas)
    dens /= np.sum(dens)
    theta_dens = np.multiply(thetas, dens)
    theta_dens /= np.sum(theta_dens)
    ax.scatter(thetas, theta_dens, c='b')
    for i, theta in enumerate(thetas):
        ax.axvline(x=theta, ymax=(theta_dens[i] / 0.25), c='b', alpha=0.5)
    ax.set_ylim((0, 0.25))
    ax.set_title('Posterior', fontweight='bold')
    ax.set_ylabel(r'$p(\theta|X)$')
    ax.set_xlabel(r'$\theta$')


fig, axes = plt.subplots(3, 1, figsize=(12, 8))
plot_discrete_prior(axes[0], thetas)
plot_discrete_likelihood(axes[1], thetas)
plot_discrete_posterior(axes[2], thetas)
plt.tight_layout()
plt.savefig('prior_likelihood_posterior.png')
plt.show()
print('here')