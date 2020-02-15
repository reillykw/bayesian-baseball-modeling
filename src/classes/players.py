
""" Player data info """
import pandas as pd
import numpy as np

players = pd.read_csv('baseballdata/People.csv')
batting = pd.read_csv('baseballdata/Batting.csv')


class PlayerData:

    def __init__(self, player_id):
        self.player_id = player_id
        self.batting = self._get_batting_data()
        self.personal = self._get_personal_data()
        self.player_name = self._get_player_name()

    def _get_batting_data(self):
        return batting.query(f"playerID == '{self.player_id}'")

    def _get_personal_data(self):
        return players.query(f"playerID == '{self.player_id}'")

    def _get_player_name(self):
        return self.personal.nameFirst + ' ' + self.personal.nameLast

    def career_hr(self):
        return self.batting.groupby('playerID')['HR'].sum().values
