# Run once at the command line
# pip install --upgrade numerapi

# python3
import os
import pandas as pd

import numerapi

DATA_DIRECTORY = os.path.expanduser("~/Dropbox/QuantFinance/Numerai/Data/")

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
current_data_path = f'{DATA_DIRECTORY}numerai_dataset_{current_round}'

tournaments = napi.get_tournaments()

for tournament in tournaments:
    tournament_predictions_exist = os.path.isfile(os.path.join(
        current_data_path,
        'predictions_' +
        tournament['name'] +
        '.csv'
        )
    )
    print(tournament['name'])
    print(tournament_predictions_exist)

# TODO:
# if tournament_predictions_exist:
#    upload

