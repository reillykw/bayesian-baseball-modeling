import src.ufuncs as utils
import pandas as pd
import scipy.stats as stats
import boto3
import os

SAMPLES = 100
BUCKET = 'bayesian-baseball'

PARAMS = {
    'tau': 10,
    'proposal_var': 5,
    'alpha_mean': 0,
    'beta_mean': 0,
    'gamma_mean': 0
}

REGION_NAME = 'us-west-2'

if __name__ == '__main__':

    test = pd.read_csv('../data/processed/test_data.csv').reset_index(drop=True)
    test['eliteStatus'] = stats.bernoulli.rvs(p=0.1, size=len(test))
    test['eliteStatusPrev'] = test.groupby('playerID')['eliteStatus'].shift(1)
    test['position_main'] = test['position_main'] - 1
    test.drop(columns=['birthYear', 'debut'], inplace=True)

    gibbs = utils.run_gibbs(samples=SAMPLES, data=test, params=PARAMS)
    # ------ When running locally use profile_name='homeusr'
    session = boto3.session.Session(region_name=REGION_NAME, profile_name='homeusr')
    utils.write_gibbs_s3(gibbs, session, bucket=BUCKET)


