import src.data.make_dataset as md
import os

if __name__ == '__main__':
    # Get all data
    print('Processing Lahman datasets...')
    all_data = md.get_player_year_data()

    # Get test data
    print('Gather test dataset...')
    additional_players = ['troutmi01', 'yelicch01', 'machama01', 'cruzne02']
    test_data = md.get_test_set(*additional_players, allow_additional_players=True)

    # send to processed folder
    print('Sending cleaned files to processed folder')
    md.send_data_to_processed(all_data)
    md.send_data_to_processed(test_data, test=True)
