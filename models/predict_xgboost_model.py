import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost

from numerai.helpers import numeric_era_converter
import numerapi

DATA_DIRECTORY = os.path.expanduser("~/numerai_data/")
napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
PATH_TO_TOURNAMENT_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_tournament_data.csv'

POSSIBLE_MARKETS = [
  'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'
]

TARGET_NAME = 'target_ken'

full_tournament_df = pd.read_csv(PATH_TO_TOURNAMENT_DATA, header = 0)

full_tournament_df['era_numeric'] = full_tournament_df['era'].apply(numeric_era_converter)

tournament_df_for_prediction = full_tournament_df.drop(['id', 'era', 'data_type'], axis=1)

dtournament = xgboost.DMatrix(tournament_df_for_prediction)

bst = pickle.load(
  open(f'{DATA_DIRECTORY}numerai_dataset_{current_round}/xgboost_{TARGET_NAME}.pickle.dat',
       'rb')
)  ,
