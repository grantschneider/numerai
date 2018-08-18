import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost

import numerapi

DATA_DIRECTORY = os.path.expanduser("~/numerai_data/")

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
PATH_TO_TRAINING_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_training_data.csv'
PATH_TO_TOURNAMENT_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_tournament_data.csv'

number_of_training_eras = current_round - 1

POSSIBLE_MARKETS = [
  'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'
]

TARGET_NAME = 'target_ken'

def _numeric_era_converter(era_string):
  return int(era_string.replace('era', ''))


def _subset_training_data_by_era(training_df, era_numeric):
  if not isinstance(era_numeric, int):
    raise ValueError(f'Expected an integer for era numeric. Instead got #{type(era_numeric)}.')
  elif 'era_numeric' not in training_df.columns:
    raise ValueError(
      f'Expected the column era_numeric in the dataframe. Instead got #{training_df.columns}.'
    )
  return training_df.loc[training_df['era_numeric'] <= era_numeric]


def _construct_X_and_Y_for_training(training_df, target_name):
  if target_name not in POSSIBLE_MARKETS:
    raise ValueError(
      f'Expected the target name to be one of #{POSSIBLE_MARKETS}. Instead got #{target_name}.'
    )
  targets_to_drop = list(set(POSSIBLE_MARKETS) - set([target_name]))
  training_df_subsetted = training_df.copy()
  training_df_subsetted.drop(['id', 'era', 'data_type', *targets_to_drop], axis=1, inplace=True)
  return training_df_subsetted.drop([target_name], axis =1), training_df_subsetted[target_name]


full_training_df = pd.read_csv(PATH_TO_TRAINING_DATA, header = 0)
full_tournament_df = pd.read_csv(PATH_TO_TOURNAMENT_DATA, header = 0)

full_training_df['era_numeric'] = full_training_df['era'].apply(_numeric_era_converter)
full_tournament_df['era_numeric'] = full_tournament_df['era'].apply(_numeric_era_converter)


# TODO: This subsetting is only necessary if we change number_of_training_eras
training_df_limited_era = _subset_training_data_by_era(
  training_df = full_training_df, era_numeric = number_of_training_eras
)

training_data_X_limited_eras, training_data_Y_limited_eras = _construct_X_and_Y_for_training(
  training_df = training_df_limited_era, target_name = TARGET_NAME
)


X_train, X_test, y_train, y_test = train_test_split(
  training_data_X_limited_eras,
  training_data_Y_limited_eras,
  test_size=0.2,
  random_state=0
)

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

gpu_res = {} # Store accuracy result

num_round = 1000
params = {'max_depth': 5,
          'learning_rate': 0.1,
          'objective': 'binary:logistic',
          #'min_child_weight': 0.1,
          #'gamma': 0.5,
          #'n_estimators': 1300,
          #'scale_pos_weight': 1,
          #'nthread': 1,
          #'tree_method': 'gpu_hist',
          #'gpu_id': 0,
          #'max_bin': 16,
          #'subsample': 0.66,
          #'colsample_bytree': 0.33
          }

bst = xgboost.train(params, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)

bst.predict(dtest)
