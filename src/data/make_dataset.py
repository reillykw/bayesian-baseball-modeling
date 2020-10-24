import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
import os

URL = "https://github.com/chadwickbureau/baseballdatabank/archive/master.zip"
DATA_FILE_PATH = '../../data/'
POSITION_DICT = {0: 'P', 1: 'C', 2: '1B', 3: '2B', 4: '3B', 5: 'SS', 6: 'LF', 7: 'CF', 8: 'RF', 9: 'DH'}


def get_lahman_data(url, out_path):
    r = requests.get(url, stream=True)
    z = ZipFile(BytesIO(r.content))
    z.extractall(out_path)
    print(f'Data extracted to {DATA_FILE_PATH}')


def get_data_sets():
    if not os.path.isdir(DATA_FILE_PATH + 'baseballdatabank-master'):
        get_lahman_data(URL, DATA_FILE_PATH)
    files = ['People.csv', 'Batting.csv', 'Appearances.csv', 'Teams.csv']
    names = ['players', 'batting', 'appearances', 'teams']
    data_sets = []
    for file in files:
        data_ = pd.read_csv(DATA_FILE_PATH + 'baseballdatabank-master/core/' + file)
        data_sets.append(data_)
    data_sets = dict(zip(names, data_sets))
    return data_sets


def batting_player_year(data):
    """
    Args
    ----
        data (pd.DataFrame): a dataframe containing batting data

    Returns
    -------
        data (pd.DataFrame): a dataframe rolled up to the player year

    Some player years have multiple records since players can be traded over
    the course of the year and may have played for multiple teams. To rectify
    this the team that corresponded with the most ABs is chosen as the `main team`.
    In the case where both records have the same number of ABs for both teams we
    check the number of games played and choose the team with the most games played.
    """

    cols = ['playerID', 'yearID', 'teamID', 'G', 'AB', 'HR']
    data = data.groupby(['playerID', 'yearID', 'teamID'], as_index=False).sum()[cols]

    # data values must be sorted by playerID and yearID
    data.sort_values(by=['playerID', 'yearID'], inplace=True)

    # get team with the most ab or games played
    team = []
    for index, player_year in data.groupby(['playerID', 'yearID']):
        n = len(player_year)

        # if there is only 1 record get the team and continue
        if n == 1:
            team.append(player_year['teamID'].values[0])
            continue

        max_g = player_year.G.max()
        max_ab = player_year.AB.max()

        # first check if the max ABs has a unique record and get the team
        # otherwise get the team with the max games played
        if len(player_year[player_year.AB == max_ab]['teamID']) > 1:
            for i in range(n):
                team.append(player_year[player_year.G == max_g]['teamID'].values[0])
        else:
            for i in range(n):
                team.append(player_year[player_year.AB == max_ab]['teamID'].values[0])

    data['team'] = team
    data = data.groupby(['playerID', 'yearID', 'team'], as_index=False).sum()

    return data[['playerID', 'yearID', 'team', 'AB', 'HR']]


def appearances_player_year(appearance_data):
    """
    Args
    ----
    appearance_data : pd.DataFrame
        Appearance data from Lahman Database
    Returns
    -------
    df
        Appearance data rolled up to player-year granularity with
        main position specified. Returns with a filter on pitchers.
    """
    cols = ['playerID', 'yearID', 'G_all', 'G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh']
    df = appearance_data.groupby(['playerID', 'yearID'], as_index=False).sum()[cols]
    df['position_main'] = main_position(df)
    return df.loc[df['position_main'] > 0, ['playerID', 'yearID', 'position_main']]


def main_position(appearance_data):
    positions = ['G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh']
    position_data = appearance_data.loc[:, appearance_data.columns.isin(positions)]
    main_pos = np.argmax(position_data.values, axis=1)
    return main_pos


def get_player_year_data():
    all_data = get_data_sets()

    # clean up batting / appearance data sets
    batting = batting_player_year(all_data['batting'])
    appearances = appearances_player_year(all_data['appearances'])

    # join all the data
    data = batting \
        .merge(appearances, on=['playerID', 'yearID'], how='inner') \
        .merge(all_data['players'][['playerID', 'nameFirst', 'nameLast', 'birthYear', 'debut']], on=['playerID'],
               how='inner') \
        .merge(all_data['teams'][['yearID', 'teamID', 'park']], left_on=['yearID', 'team'],
               right_on=['yearID', 'teamID'])

    # add additional columns
    data['playerAge'] = data['yearID'] - data['birthYear']
    data['yearsInMLB'] = data.groupby('playerID')['yearID'].rank(method='dense')
    data['debut'] = pd.to_datetime(data['debut'])
    data['playerFullName'] = data['nameFirst'] + ' ' + data['nameLast']

    # filter data to >= year 2000
    # TODO: Get rid of this. This will skew our data especially in the early years
    data = data[data['debut'].dt.year >= 2000].sort_values(['playerID', 'yearID'])

    # Specify data types
    data = data.astype({'birthYear': np.int64, 'playerAge': np.int64, 'yearsInMLB': np.int64})

    # drop
    data.drop(columns=['teamID', 'nameFirst', 'nameLast'], inplace=True)

    return data


def get_test_set(*args, size=20, random_seed=0, allow_additional_players=False):
    """
    """
    np.random.seed(random_seed)
    py_data = get_player_year_data()

    all_players = py_data['playerID'].unique()
    n_players = len(all_players)
    r_idx = np.random.randint(0, n_players, size=size)
    r_players = all_players[r_idx]

    if allow_additional_players:
        for player in args:
            r_players = np.append(r_players, player)

    test_data = py_data.loc[py_data['playerID'].isin(r_players)]

    return test_data


def send_data_to_processed(data, test=False):
    if test:
        data.to_csv(DATA_FILE_PATH + 'processed/' + 'test_data.csv', index_label=False)
    else:
        data.to_csv(DATA_FILE_PATH + 'processed/' + 'full_data.csv', index_label=False)
