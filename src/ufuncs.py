import numpy as np
import scipy.stats as stats
import datetime as dt
import boto3
import pickle


def theta(alpha, beta, age_trajectory_val):
    inner = alpha + beta + age_trajectory_val
    if inner > 100:
        return 1 - 1e-16
    if inner < -100:
        return 1e-16
    return 1 / (1 + np.exp(-inner))


def logit_theta(theta):
    return np.log(theta/(1-theta))


def age_trajectory(age, gamma):
    """ Put docstring here """
    # return gamma[0] + age * gamma[1] + (age ** 2) * gamma[2] + (age ** 3) * gamma[3]
    if age < 30:
        return gamma[0] + age * gamma[1]
    else:
        return gamma[2] + age * gamma[3]


def log_prior(alpha, beta, gamma, nu, omega, tau):
    """ Need Docstring """

    psum = 9*np.log(2)
    psum += sum([stats.multivariate_normal(0, tau).logpdf(alpha_arr).sum() for alpha_arr in alpha])
    psum += stats.norm(0, tau).logpdf(beta).sum()
    psum += sum([stats.norm(0, tau).logpdf(gamma_arr).sum() for gamma_arr in gamma])

    # I'm gonna have to mess with this one
    for pos_vals, pos_params in zip(nu, omega):
        for vals, params in zip(pos_vals, pos_params):
            psum += stats.dirichlet.logpdf(vals, params)

    return psum


def log_elite_transitions(pos, prev, curr, nu_params):
    if np.isnan(prev):
        prev = 0
    return np.log(nu_params[pos, int(prev), int(curr)])


def log_posterior(alpha, beta, gamma, nu, omega, tau, data):
    """
    Args
    ----
    alpha : np.array(1, 2)
        Intercept coefficients
    beta : np.array(1, 1)
        Park coefficients
    gamma: np.array(1, 4)
        Spline coefficients
    nu : np.array(2, 2)
        Transition probabilities
    omega : np.array(2, 2)
        Dirichlet parameters
    tau : int
        Variance for normal priors
    data
        All Player Data

    Returns
    -------
        Log Likelihood of the posterior distribution
    """

    post_sum = 0
    parks = sorted(data['park'].unique())

    for _, row in data.iterrows():
        # 2nd equation on RHS equation (8) return the LOG of p(θ)
        age = row.playerAge
        pos = row.position_main
        park = row.park
        hrs = row.HR
        ab = row.AB
        es_prev = row.eliteStatusPrev
        es_curr = row.eliteStatus

        age_traj = age_trajectory(age, gamma[pos])

        park_idx = parks.index(park)

        rate = theta(alpha[pos, es_curr], beta[park_idx], age_traj)
        log_theta = np.log(rate)

        # capture rates > 1 and give them .99 for now
        rate_trunc = rate if rate < 1 else 0.99

        # 1st equation on RHS of equation(8)
        post_sum += stats.binom(n=ab, p=rate_trunc).logpmf(k=hrs)
        # 2nd equation on RHS of equation(8)
        post_sum += log_theta
        # 3rd equation on RHS of equation(8)
        post_sum += log_elite_transitions(pos, es_prev, es_curr, nu)

    # 4th equation on the RHS of equation(8)
    post_sum += log_prior(alpha, beta, gamma, nu, omega, tau)

    return post_sum


def transitions(players, curr_params):
    for index, player in players:
        elite_status = player[['eliteStatus', 'eliteStatusPrev']].values
        for curr, prev in elite_status:
            curr = int(curr)
            prev = int(prev) if ~np.isnan(prev) else prev
            if np.isnan(prev):
                continue
            curr_params[prev][curr] += 1
    return curr_params


def update_omegas(data, position, curr_params):
    curr_param_set = curr_params[position, :, :]
    pos = data[data.position_main == position]
    players = pos.groupby('playerID')
    return transitions(players, curr_param_set)


def update_nus(nu, omega):
    pos, rows, cols = nu.shape
    for i in range(pos):
        for row in range(rows):
            nu[i, row, :] = stats.dirichlet.rvs(alpha=omega[i, row, :])
    return nu


def forward_summing(P, player, alpha, beta, gamma):
    """
    Args
    ----
    P : np.array
        position specific elite transition matrix
    player : pd.DataFrame.GroupBy
        Player-year groupby pandas dataframe
    alpha : np.array
        position specific intercepts
    beta : np.array
        park specific coef
    gamma : np.array
        position specific spline coefs

    Returns
    ------
    M
        a transition matrix thingy
    """

    # Get the position for the first year for the player
    pos_0 = player.position_main[0]
    # Initial transition matrix based on position
    init_P = P[pos_0, :, :]

    years = len(player)

    pi = np.array([init_P[0, 1], init_P[1, 0]])

    M = np.zeros((years, 2))
    M[0, :] = pi

    for yr in range(1, years):
        pos_n = player.position_main[yr]
        new_P = P[pos_n, :, :]

        Rtemp = M[yr - 1, :].dot(new_P).reshape(1, -1)

        age = player.playerAge[yr]
        ab = player.AB[yr]
        hrs = player.HR[yr]

        age_traj_val = age_trajectory(age, gamma[pos_n])
        theta0 = theta(alpha[pos_n, 0], beta[pos_n], age_traj_val)
        theta1 = theta(alpha[pos_n, 1], beta[pos_n], age_traj_val)

        p0 = stats.binom(n=ab, p=theta0).pmf(k=hrs)
        p1 = stats.binom(n=ab, p=theta1).pmf(k=hrs)

        # does it make sense to make these values arbitrarily small
        # should something else happen here?
        if p0 == 0:
            p0 = 1e-32

        if p1 == 0:
            p1 = 1e-32

        Rtemp *= np.array([p0, p1])
        Rtemp /= Rtemp.sum()

        M[yr, :] = Rtemp

    return M


def back_sampling(P, M, player):
    """
    Args
    ----
    P : np.array
        Position specific transition matrices
    M : np.array
        Matrix from forward summing function
    player : pd.DataFrame.GroupBy
        Player-year groupby pandas dataframe
    Returns
    -------

    """
    # Get the position for the first year for the player
    pos_0 = player.position_main[0]
    # Initial transition matrix based on position
    init_P = P[pos_0, :, :]

    years = len(player)

    prob_e_last = M[years - 1, 1]
    last_e = stats.bernoulli(p=prob_e_last).rvs()

    samples = [last_e]
    for yr in range(years - 1, 0, -1):
        pos_n = player.position_main[yr]
        new_P = P[pos_n, :, :]

        e = last_e
        Rtemp = M[yr, :]
        Rtemp *= np.array([new_P[0, e], new_P[1, e]])
        Rtemp /= sum(Rtemp)

        last_e = stats.bernoulli(p=Rtemp[1]).rvs()
        samples.append(last_e)

    return np.array(list(reversed(samples)))


def initialize_params(data, params):

    tau = params['tau']
    alpha_mean = params['alpha_mean']
    beta_mean = params['beta_mean']
    gamma_mean = params['gamma_mean']

    alpha = stats.multivariate_normal.rvs(mean=alpha_mean, cov=tau, size=(9, 2))
    # alpha params must be sorted such that alpha[0] < alpha[1]
    alpha.sort()
    beta = stats.norm.rvs(loc=beta_mean, scale=tau, size=len(data.park.unique()))
    gamma = stats.norm.rvs(loc=gamma_mean, scale=tau, size=36).reshape(9, 4)
    nu = stats.dirichlet.rvs(alpha=np.array([1, 1]), size=18).reshape(9, 2, 2)
    omega = np.ones((9, 2, 2))
    elite = data.eliteStatus.values

    return alpha, beta, gamma, nu, omega, elite


def gibbs_sampler(samples, data, params):

    tau = params['tau']
    prop_var = params['proposal_var']

    # Proposal distribution for movements
    proposal_distribution = stats.norm(loc=0, scale=prop_var)

    ## alpha = stats.multivariate_normal.rvs(mean=0, cov=tau, size=(9, 2))
    # alpha = stats.multivariate_normal.rvs(mean=alpha_mean, cov=tau, size=(9, 2))
    # # alpha params must be sorted such that alpha[0] < alpha[1]
    # alpha.sort()
    #
    # beta = stats.norm.rvs(loc=beta_mean, scale=tau, size=len(data.park.unique()))
    # gamma = stats.norm.rvs(loc=gamma_mean, scale=tau, size=36).reshape(9, 4)
    #
    # nu = stats.dirichlet.rvs(alpha=np.array([1, 1]), size=18).reshape(9, 2, 2)
    # omega = np.ones((9, 2, 2))
    # elite = data.eliteStatus.values

    alpha, beta, gamma, nu, omega, elite = initialize_params(data, params)

    alpha_samples = [alpha]
    beta_samples = [beta]
    gamma_samples = [gamma]
    nu_samples = [nu]
    # elite_samples = [elite]

    alpha_accept = 0
    beta_accept = 0
    gamma_accept = 0

    for n in range(samples):
        # alpha update
        for i, alpha_ in enumerate(alpha):
            # propose new position alpha
            alpha_c = alpha.copy()
            prop_a = alpha_ + proposal_distribution.rvs(size=2)
            prop_a.sort()
            # insert new
            alpha_c[i, :] = prop_a
            rho_a = log_posterior(alpha_c, beta, gamma, nu, omega, tau, data) - \
                log_posterior(alpha, beta, gamma, nu, omega, tau, data)
            if np.isnan(rho_a):
                rho_a = 0
            rho_a = min(0, rho_a)
            accept = np.random.rand()
            if rho_a > np.log(accept):
                alpha_accept += 1
                alpha = alpha_c

        alpha_samples.append(alpha)


        # alpha_prop = alpha + proposal_distribution.rvs(size=(9, 2))
        # alpha_prop.sort()
        # rho_a = log_posterior(alpha_prop, beta, gamma, nu, omega, tau, data) - \
        #     log_posterior(alpha, beta, gamma, nu, omega, tau, data)
        # if np.isnan(rho_a):
        #     rho_a = 0
        # rho_a = min(0, rho_a)
        # accept = np.random.rand()
        # if rho_a > np.log(accept):
        #     alpha_accept += 1
        #     alpha = alpha_prop
        # alpha_samples.append(alpha)

        for i, beta_ in enumerate(beta):
            beta_c = beta.copy()
            prop_b = beta_ + proposal_distribution.rvs()
            beta_c[i] = prop_b
            rho_b = log_posterior(alpha, beta_c, gamma, nu, omega, tau, data) - \
                log_posterior(alpha, beta, gamma, nu, omega, tau, data)
            if np.isnan(rho_b):
                rho_b = 0
            rho_b = min(0, rho_b)
            accept = np.random.rand()
            if rho_b > np.log(accept):
                beta_accept += 1
                beta = beta_c

        beta_samples.append(beta)


        # beta update
        # beta_prop = beta + proposal_distribution.rvs(size=len(parks))
        # rho_b = log_posterior(alpha, beta_prop, gamma, nu, omega, tau, data) - \
        #     log_posterior(alpha, beta, gamma, nu, omega, tau, data)
        # if np.isnan(rho_b):
        #     rho_b = 0
        # rho_b = min(0, rho_b)
        # accept = np.random.rand()
        # if rho_b > np.log(accept):
        #     beta_accept += 1
        #     beta = beta_prop
        # beta_samples.append(beta)

        for i, gamma_ in enumerate(gamma):
            gamma_c = gamma.copy()
            prop_g = gamma_ + proposal_distribution.rvs(size=4)
            gamma_c[i, :] = prop_g
            rho_g = log_posterior(alpha, beta, gamma_c, nu, omega, tau, data) - \
                log_posterior(alpha, beta, gamma, nu, omega, tau, data)
            if np.isnan(rho_g):
                rho_g = 0
            rho_g = min(0, rho_g)
            accept = np.random.rand()
            if rho_g > np.log(accept):
                gamma_accept += 1
                gamma = gamma_c

        gamma_samples.append(gamma)

        # gamma update
        # gamma_prop = gamma + proposal_distribution.rvs(size=36).reshape(9, 4)
        # rho_g = log_posterior(alpha, beta, gamma_prop, nu, omega, tau, data) - \
        #     log_posterior(alpha, beta, gamma, nu, omega, tau, data)
        # if np.isnan(rho_g):
        #     rho_g = 0
        # rho_g = min(0, rho_g)
        # accept = np.random.rand()
        # if rho_g > np.log(accept):
        #     gamma_accept += 1
        #     gamma = gamma_prop
        # gamma_samples.append(gamma)

        # omega update
        omega = np.array([update_omegas(data=data, position=i, curr_params=omega) for i in range(9)])
        # nu update
        nu = update_nus(nu, omega)
        nu_samples.append(nu)

        # breakpoint()
        # elite status update
        elite_status_update = []
        for index, player in data.groupby('playerID'):
            player.reset_index(drop=True, inplace=True)
            forward_sum = forward_summing(nu, player, alpha, beta, gamma)
            back_samp = back_sampling(nu, forward_sum, player)
            elite_status_update.extend(back_samp)

        elite = elite_status_update

        if n % 10 == 0:
            print(n)

    return (alpha_samples, alpha_accept), (beta_samples, beta_accept), (gamma_samples, gamma_accept), nu_samples, elite


def run_gibbs(data, samples, params):
    import time
    start = time.time()
    gibbs = gibbs_sampler(samples=samples, data=data, params=params)
    end = time.time()

    def get_hms(seconds):
        hours_ = seconds // 3600
        minutes_ = (seconds % 3600) // 60
        seconds_ = (seconds % 3600) % 60
        return str(round(hours_)) + ':' + str(round(minutes_)) + ':' + str(round(seconds_))

    print(f'Gibbs sampler took {get_hms(end - start)}')
    return gibbs


def write_gibbs_s3(gibbs_data, session, bucket):
    s3 = session.resource('s3')
    s3_client = session.client('s3')

    date_time_str = dt.datetime.now().strftime(format='%Y%m%d')
    s3_client.put_object(Bucket=bucket, Key=date_time_str + '/')

    file_names = ['alpha_samples.pkl', 'beta_samples.pkl', 'gamma_samples.pkl',  'nu_samples.pkl', 'elite_samples.pkl']
    for i, file in enumerate(file_names):
        pickle_obj = pickle.dumps(gibbs_data[i])
        s3.Object(bucket, date_time_str + '/' + file).put(Body=pickle_obj)

    print(f'Efficiency for alpha sampling {gibbs_data[0][1] / (len(gibbs_data[0][0]) * 9)}')
    print(f'Efficiency for beta sampling {gibbs_data[1][1] / (len(gibbs_data[1][0]) * 63)}')
    print(f'Efficiency for gamma sampling {gibbs_data[2][1] / (len(gibbs_data[2][0]) * 9)}')
    print('Sending pickled data to s3...')
