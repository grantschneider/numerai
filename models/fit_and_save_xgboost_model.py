import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost

# If this doesn't work you probably need to add the following line to the end of your ~/.bashrc
# export PYTHONPATH="${PYTHONPATH}:~/numerai/"
from numerai.helpers import DATA_DIRECTORY, numeric_era_converter, POSSIBLE_MARKET_NAMES
import numerapi

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
PATH_TO_TRAINING_DATA = f'{DATA_DIRECTORY}numerai_dataset_{current_round}/numerai_training_data.csv'

number_of_training_eras = current_round - 1

TARGET_NAME = 'target_charles'
HYPERPARAMETER_OPTIMIZE = False


def _subset_training_data_by_era(training_df, era_numeric):
  if not isinstance(era_numeric, int):
    raise ValueError(f'Expected an integer for era numeric. Instead got #{type(era_numeric)}.')
  elif 'era_numeric' not in training_df.columns:
    raise ValueError(
      f'Expected the column era_numeric in the dataframe. Instead got #{training_df.columns}.'
    )
  return training_df.loc[training_df['era_numeric'] <= era_numeric]


def _construct_X_and_Y_for_training(training_df, target_name):
  if target_name not in POSSIBLE_MARKET_NAMES:
    raise ValueError(
      f'Expected the target name to be one of #{POSSIBLE_MARKET_NAMES}. Instead got #{target_name}.'
    )
  targets_to_drop = list(set(POSSIBLE_MARKET_NAMES) - set([target_name]))
  training_df_subsetted = training_df.copy()
  training_df_subsetted.drop(['id', 'era', 'data_type', *targets_to_drop], axis=1, inplace=True)
  return training_df_subsetted.drop([target_name], axis =1), training_df_subsetted[target_name]


full_training_df = pd.read_csv(PATH_TO_TRAINING_DATA, header = 0)

full_training_df['era_numeric'] = full_training_df['era'].apply(numeric_era_converter)


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

# TODO: Add checks for how good the model is

# TODO: move hyperparameter tuning below to a different file
# TODO: Follow https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
gridsearch_params = [
    (max_depth, min_child_weight, learning_rate, gamma)
    for max_depth in range(6,11)
    for min_child_weight in range(6,11)
    for learning_rate in np.linspace(0, 0.05, num=6)
    for gamma in [0]#np.linspace(0, 6, num=3)
]


if HYPERPARAMETER_OPTIMIZE:
  params_start = {
            'objective': 'binary:logistic',
            #'n_estimators': 1300,
            #'scale_pos_weight': 1,
            #'nthread': 1,
            'tree_method': 'gpu_hist', # comment this line out if not running xgboost on gpu
            #'gpu_id': 0,
            #'max_bin': 16,
            #'subsample': 0.66,
            #'colsample_bytree': 0.33
            }
  # Find best params and log loss
  min_ll = float("Inf")
  best_params = None
  for max_depth, min_child_weight, learning_rate, gamma in gridsearch_params:
      print("CV with max_depth={}, min_child_weight={}, learning_rate={}, gamma={}".format(
                               max_depth, min_child_weight, learning_rate, gamma
                               ))
      # Update our parameters
      params_start['max_depth'] = max_depth
      params_start['min_child_weight'] = min_child_weight
      params_start['learning_rate'] = learning_rate
      params_start['gamma'] = gamma
      # Run CV
      cv_results = xgboost.cv(
          params_start,
          dtrain,
          num_boost_round=999,
          seed=42,
          nfold=5,
          metrics={'auc', 'logloss'},
          early_stopping_rounds=10
      )
      # Update best log loss
      mean_ll = cv_results['test-logloss-mean'].min()
      boost_rounds = cv_results['test-logloss-mean'].argmin()
      print("\tlog loss {} for {} rounds".format(mean_ll, boost_rounds))
      if mean_ll < min_ll:
          min_ll = mean_ll
          best_params = (max_depth, min_child_weight, learning_rate, gamma)
  print("Best params: {}, {}, {}, {} log-loss: {}".format(
    best_params[0], best_params[1], best_params[2], best_params[3],
    min_ll))

num_round = 288
results = {}
params_optimal = {'max_depth': 6,
          'learning_rate': 0.04,
          'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'min_child_weight': 10,
          'gamma': 0,
          #'n_estimators': 1300,
          #'scale_pos_weight': 1,
          #'nthread': 1,
          'tree_method': 'gpu_hist', # comment this line out if not running xgboost on gpu
          #'gpu_id': 0,
          #'max_bin': 16,
          #'subsample': 0.66,
          #'colsample_bytree': 0.33
          }

bst = xgboost.train(
  params_optimal,
  dtrain,
  num_round,
  evals=[(dtest, 'test')],
  evals_result=results
)

mean_test = np.mean(y_test)
baseline_predictions = np.ones(y_test.shape) * mean_test
bst_predictions = bst.predict(dtest)

ll_baseline = log_loss(y_test, baseline_predictions)
ll_bst = log_loss(y_test, bst_predictions)

print(ll_baseline)
print(ll_bst)

pickle.dump(
  bst,
  open(f'{DATA_DIRECTORY}numerai_dataset_{current_round}/xgboost_{TARGET_NAME}.pickle.dat',
       'wb')
)

