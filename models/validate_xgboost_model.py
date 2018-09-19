# TODO: Make this script share more logic with predict_xgboost_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost

from numerai.helpers import DATA_DIRECTORY, numeric_era_converter, POSSIBLE_MARKET_NAMES
import numerapi

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
PATH_TO_TOURNAMENT_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_tournament_data.csv'

TARGET_NAME = 'target_charles'

full_tournament_df = pd.read_csv(PATH_TO_TOURNAMENT_DATA, header = 0)

full_tournament_df['era_numeric'] = full_tournament_df['era'].apply(numeric_era_converter)

ids = full_tournament_df['id']


full_tournament_df = full_tournament_df
full_tournament_df.drop(full_tournament_df[full_tournament_df.data_type != 'validation'].index, inplace=True)
y_validate = full_tournament_df[TARGET_NAME]

tournament_df_for_validation = full_tournament_df.drop(['id', 'era', 'data_type', *POSSIBLE_MARKET_NAMES], axis=1)

dtournament = xgboost.DMatrix(tournament_df_for_validation)

bst = pickle.load(
  open(f'{DATA_DIRECTORY}numerai_dataset_{current_round}/xgboost_{TARGET_NAME}.pickle.dat',
       'rb')
)

validation_predictions = bst.predict(dtournament)
ll_validate = log_loss(y_validate, validation_predictions)
print(ll_validate)
