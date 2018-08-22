import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost

from numerai.helpers import DATA_DIRECTORY, numeric_era_converter, POSSIBLE_MARKET_NAMES
import numerapi

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
PATH_TO_TOURNAMENT_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_tournament_data.csv'

TARGET_NAME = 'target_bernie'

full_tournament_df = pd.read_csv(PATH_TO_TOURNAMENT_DATA, header = 0)

full_tournament_df['era_numeric'] = full_tournament_df['era'].apply(numeric_era_converter)

ids = full_tournament_df['id']
tournament_df_for_prediction = full_tournament_df.drop(['id', 'era', 'data_type', *POSSIBLE_MARKET_NAMES], axis=1)

dtournament = xgboost.DMatrix(tournament_df_for_prediction)

bst = pickle.load(
  open(f'{DATA_DIRECTORY}numerai_dataset_{current_round}/xgboost_{TARGET_NAME}.pickle.dat',
       'rb')
)

predictions = bst.predict(dtournament)
results_df = pd.DataFrame(data={'probability_bernie':predictions})
joined = pd.DataFrame(ids).join(results_df)

joined.to_csv(
  f'{DATA_DIRECTORY}numerai_dataset_{current_round}/bernie_submission.csv',
  index=False
)
